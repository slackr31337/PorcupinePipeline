"""
Run Porcupine wake word listener 
and send audio to Home Assistant audio pipeline
"""
from __future__ import annotations

import os
import sys
import time
import struct
import signal
import logging
import argparse
import asyncio
import audioop
import ssl
import threading
import requests
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from asyncio import Future
import warnings

import aiohttp
import pvporcupine
from pvporcupine import Porcupine
from pvrecorder import PvRecorder
import simpleaudio

from cli_args import get_cli_args

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.raiseExceptions = False

_LOGGER = logging.getLogger(__name__)

RESULT = "result"
ERROR = "error"
MESSAGE = "message"
EVENT = "event"
NAME = "name"
DATA = "data"
TYPE = "type"
ID = "id"

WEBSOCKET_TIMEOUT = 10  # seconds


##########################################
@dataclass
class State:
    """Client state"""

    args: argparse.Namespace
    connected: bool = False
    running: bool = True
    recording: bool = False
    audio_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)


##########################################
class HAConnection:
    """
    Class handling all the low level websocket communication with HA.
    
    Clients should only use the 3 high-level public functions for communicating with HA:
        send_and_receive_json(message):  sends JSON message and receives the response
        receive_json(message_id):        receives JSON message with a specific message_id
        send_bytes(bytes):               sends binary message (without response)
    
    Responses are properly dispatched based on their message_id. So the correct response
    will always be received, even if messages arrive in different order. Messages are
    also queued, so receive_json() will succeed even if the message arrived before its call.
    """

    def __init__(self, state: State, websocket_url):
        self.__state = state
        self.__websocket_url = websocket_url
        self.__message_id = 1
        self.__msg_futures: Dict[int,Future] = {}     # message_id => future of receive_json()
        self.__msg_queues: Dict[int,List[dict]] = {}  # message_id => list of messages

        __conn = aiohttp.TCPConnector()
        self.__session = aiohttp.ClientSession(connector=__conn)

        sslcontext = None
        if websocket_url.startswith("wss"):
            sslcontext = ssl.create_default_context(
                purpose=ssl.Purpose.CLIENT_AUTH
            )

        self.__websocket_context = self.__session.ws_connect(
            websocket_url,
            ssl=sslcontext,
            timeout=WEBSOCKET_TIMEOUT,
        )

    # Async context manager
    async def __aenter__(self):
        await self.__session.__aenter__()
        self.__websocket = await self.__websocket_context.__aenter__()
        self.__receive_loop_task = asyncio.create_task(self.__receive_loop())

        await self.__authenticate()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.__session.__aexit__(exc_type, exc, tb)
        await self.__websocket_context.__aexit__(exc_type, exc, tb)

        self.__receive_loop_task.cancel()

    async def __receive_loop(self) -> None:
        """Loop that receives and dispatches messages."""

        try:
            # Run until the task is cancelled
            while True:
                try:
                    msg = await self.__websocket.receive_json(timeout=WEBSOCKET_TIMEOUT)
                except asyncio.TimeoutError:
                    continue

                # fulfill future, if available, otherwise queue the message
                message_id = msg.get('id', 0)      # can be None for auth messages
                future = self.__msg_futures.pop(message_id, None)
                if future:
                    future.set_result(msg)
                else:
                    self.__msg_queues.setdefault(message_id, []).append(msg)

        except asyncio.CancelledError as e:
            _LOGGER.debug("WS receive loop finished")

    async def __authenticate(self) -> None:
        """Authenticate websocket connection to HA"""

        _LOGGER.info("Authenticating to: %s", self.__websocket_url)

        self.__state.connected = False

        msg = await self.receive_json(message_id=0)
        assert msg[TYPE] == "auth_required", msg

        # raw send, no message id
        await self.__websocket.send_json(
            {
                TYPE: "auth",
                "access_token": self.__state.args.token,
            }
        )

        msg = await self.receive_json(message_id=0)
        assert msg.get(TYPE) == "auth_ok", msg
        _LOGGER.info(
            "Authenticated to Home Assistant version %s", msg.get("ha_version")
        )
        self.__state.connected = True


    ### Public functions to communicate with HA #############################33

    async def send_and_receive_json(self, message: dict) -> dict:
        """Send JSON message and receives the response"""

        assert isinstance(message, dict), "Invalid WS message type"

        assert self.__state.connected, "WS not connected"

        message[ID] = self.__message_id
        self.__message_id += 1

        _LOGGER.debug("send_json() message=%s", message)

        await self.__websocket.send_json(message)

        response = await self.receive_json(message[ID])
        _LOGGER.debug("send_json() response=%s", response)
        return response

    async def send_bytes(self, bts: bytes):
        """Send binary message (without response)"""

        await self.__websocket.send_bytes(bts)

    def receive_json(self, message_id: int) -> Future[dict]:
        """Receive JSON message with a specific (previously created) message_id"""

        # We return a future, which is fulfilled either now or later.
        future = asyncio.get_running_loop().create_future()

        queue = self.__msg_queues.get(message_id)
        if queue:
            # There is already a queued message we fulfill the future immediately.
            future.set_result(queue.pop(0))
            if not queue:
                del self.__msg_queues[message_id]
        else:
            # No message yet, we store the future to be later fulfilled by receive_loop().
            # To simplify dispatch, it is assumed that at most one active
            # receive_json call exists for each message_id.
            assert message_id not in self.__msg_futures, f"receive_json already active for message_id {message_id}"

            self.__msg_futures[message_id] = future

        return future
 

##########################################
class PorcupinePipeline:
    """Class used to process audio pipeline using HA websocket"""

    websocket_url: str
    _ha_connection: HAConnection
    _ha_url: str
    _last_ping = 0
    _recorder: PvRecorder
    _porcupine: Porcupine
    _devices: Dict[int,str] = {}
    _conversation_id: int
    _followup = False

    ##########################################
    def __init__(self, args: argparse.Namespace):
        """Setup Websocket client and audio pipeline"""

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        self._state = State(args=args)
        self._state.running = False

        self._event_loop = asyncio.get_event_loop()

        for idx, device in enumerate(PvRecorder.get_audio_devices()):
            self._devices[idx] = device
            _LOGGER.info("Device %d: %s", idx, device)

        _id = args.audio_device
        audio_device = self._devices.get(_id)
        if not audio_device:
            _LOGGER.error("Invalid audio device id: %s", _id)
            return None

        _LOGGER.info("Using Device %d: %s", _id, audio_device)
        if args.show_audio_devices:
            sys.exit(0)

        self._porcupine = get_porcupine(self._state)
        self._recorder = PvRecorder(
            device_index=args.audio_device,
            frame_length=self._porcupine.frame_length,
        )

        self._audio_thread = threading.Thread(
            target=self.read_audio,
            daemon=True,
        )
        self._setup_urls()

    ##########################################
    def _setup_urls(self) -> None:
        """Setup Home-Assistant and Websocket URLs"""

        server = self._state.args.server
        port = self._state.args.server_port
        proto = "http"

        self.websocket_url = "ws"
        if self._state.args.server_https:
            proto += "s"
            self.websocket_url = proto

        self._ha_url = f"{proto}://{server}:{port}"
        self.websocket_url += f"://{server}:{port}/api/websocket"

    ##########################################
    def start(self) -> None:
        """Start listening for wake word"""

        self._state.running = True

        _LOGGER.info("Starting audio listener thread")
        self._audio_thread.start()

        self._event_loop.run_until_complete(self._start_audio_pipeline())

    ##########################################
    def stop(self, signum=0, frame=None) -> None:
        """Stop audio thread and loop"""

        _LOGGER.info("Stopping")
        if signum:
            signame = signal.Signals(signum).name
            _LOGGER.debug(signame)

        self._state.running = False
        self._state.recording = False
        self._recorder.stop()

        self._audio_thread.join(1)

        if hasattr(self._porcupine, "delete"):
            self._porcupine.delete()

        del self._porcupine
        del self._ha_connection
        sys.exit(0)

    ##########################################
    async def _ping(self):
        """Send Ping to HA"""

        if not self._state.running or not self._state.connected:
            return

        now = int(time.time())
        if now - self._last_ping < 30:
            await asyncio.sleep(0.3)
            return

        response = await self._ha_connection.send_and_receive_json({TYPE: "ping"})
        if response.get(TYPE) == "pong":
            self._state.connected = True

        else:
            self._state.connected = False
            _LOGGER.error(response)

        self._last_ping = int(time.time())

    ##########################################
    def _disconnect(self) -> None:
        """Websocket disconnect callback"""

        self._state.connected = False

    ##########################################
    async def _start_audio_pipeline(self):
        """Start HA audio pipeline"""

        _LOGGER.info("Starting audio pipeline loop")

        async with HAConnection(self._state, self.websocket_url) as self._ha_connection:
            await self.get_audio_pipeline()
            await self._process_loop()

    ##########################################
    async def get_audio_pipeline(self) -> None:
        """Return ID of audio pipeline"""

        self._pipeline_id: Optional[str] = None
        if self._state.args.pipeline:
            _LOGGER.info(
                "Using Home Assistant audio pipeline %s", self._state.args.pipeline
            )

            # Get list of available pipelines and resolve name
            msg = await self._ha_connection.send_and_receive_json(
                {
                    TYPE: "assist_pipeline/pipeline/list",
                }
            )
            _LOGGER.debug(msg)
            if RESULT not in msg:
                _LOGGER.error("FAiled to get audio pipeline from HA")
                _LOGGER.error(msg)
                return

            pipelines = msg[RESULT]["pipelines"]
            for pipeline in pipelines:
                if pipeline[NAME] == self._state.args.pipeline:
                    self._pipeline_id = pipeline.get(ID)
                    break

            if not self._pipeline_id:
                raise ValueError(
                    f"No pipeline named {self._state.args.pipeline} in {pipelines}"
                )

    ##########################################
    async def _process_loop(self) -> None:
        """Process audio and wake word events"""

        _LOGGER.info("Starting audio processing loop")
        while self._state.running:
            # Clear audio queue
            while not self._state.audio_queue.empty():
                self._state.audio_queue.get_nowait()

            _LOGGER.info("Waiting for wake word to trigger audio")
            while not self._state.recording:
                await self._ping()

            # Run audio pipeline
            pipeline_args = {
                TYPE: "assist_pipeline/run",
                "start_stage": "stt",
                "end_stage": "tts",
                "input": {
                    "sample_rate": 16000,
                },
            }
            if self._pipeline_id:
                pipeline_args["pipeline"] = self._pipeline_id

            # Send audio pipeline args to HA
            msg = await self._ha_connection.send_and_receive_json(pipeline_args)
            if not msg.get("success"):
                _LOGGER.error(
                    msg.get(ERROR, {}).get(MESSAGE, "Pipeline failed to start")
                )
                return

            _LOGGER.info(
                "Listening and sending audio to voice pipeline %s", self._pipeline_id
            )
            await self.stt_task(msg[ID])

    ##########################################
    async def stt_task(self, message_id) -> None:
        """
        Create task to process speech to text.

        message_id: The message id used to call assist_pipeline/run.
                    Ensures that we only read events of that pipeline.
        """

        # Audio loop for single pipeline run
        count = 0

        # Get handler id.
        # This is a single byte prefix that needs to be in every binary payload.
        msg = await self._ha_connection.receive_json(message_id)
        _LOGGER.debug(msg)

        handler_id = bytes(
            [msg[EVENT][DATA]["runner_data"].get("stt_binary_handler_id")]
        )

        receive_event_future = self._ha_connection.receive_json(message_id)

        while self._state.connected:
            audio_chunk = await self._state.audio_queue.get()
            if not audio_chunk:
                break

            # Prefix binary message with handler id
            send_audio_task = asyncio.create_task(
                self._ha_connection.send_bytes(handler_id + audio_chunk)
            )
            pending = {send_audio_task, receive_event_future}
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receive_event_future in done:
                # the only messages reiceived on our message_id should be events
                event = receive_event_future.result()
                assert EVENT in event
                event_type = event[EVENT].get(TYPE)
                event_data = event[EVENT].get(DATA)

                if event_type == "run-end":
                    count += 1
                    _LOGGER.debug("[%s] Pipeline finished", count)
                    if self._state.args.follow_up and count < 4:
                        self._state.recording = True
                    else:
                        self._state.recording = False
                    break

                if event_type == "error":
                    self._state.recording = False
                    _LOGGER.info(
                        "%s. Listening stopped",
                        event_data.get("message"),
                    )
                    break

                elif event_type == "intent-end":
                    intent = event_data.get("intent_output", {})
                    self._conversation_id = intent.get("conversation_id")

                elif event_type == "stt-end":
                    # HA finished processing speech to text with result
                    speech = event_data["stt_output"].get("text")
                    _LOGGER.info("Recognized speech: %s", speech)

                elif event_type == "tts-end":
                    # URL of text to speech audio response (relative to server)
                    tts_url = self._ha_url
                    tts_url += event_data["tts_output"].get("url")
                    _LOGGER.info("Play response: %s", tts_url)
                    await self._play_response(tts_url)

                elif event_type == "stt-start":
                    # HA has started processing speech to text
                    _LOGGER.debug("HA stt using %s", event_data.get("engine"))

                else:
                    _LOGGER.debug("event_type=%s", event_type)
                    _LOGGER.debug("event=%s", event)
                    # _LOGGER.debug("event_data=%s", event_data)

                receive_event_future = self._ha_connection.receive_json(message_id)

            if not self._state.running:
                break

            if send_audio_task not in done:
                await send_audio_task

    ##########################################
    def read_audio(self) -> None:
        """Reads chunks of raw audio from standard input."""
        try:
            args = self._state.args
            keywords = args.keywords
            ratecv_state = None

            _LOGGER.debug("Reading audio")
            self._recorder.start()
            self._state.recording = False
            while self._state.running:
                try:
                    pcm = self._recorder.read()

                except OSError as err:
                    _LOGGER.error("Exception: %s", err)
                    self._state.running = False
                    break

                if not self._state.recording:
                    result = self._porcupine.process(pcm)

                    if result >= 0:
                        _LOGGER.info("Detected keyword `%s`", keywords[result])
                        self._state.recording = True

                if self._state.recording:
                    chunk = struct.pack("h" * len(pcm), *pcm)

                    # Convert to 16Khz, 16-bit, mono
                    if args.channels != 1:
                        chunk = audioop.tomono(chunk, args.width, 1.0, 1.0)

                    if args.width != 2:
                        chunk = audioop.lin2lin(chunk, args.width, 2)

                    if args.rate != 16000:
                        chunk, ratecv_state = audioop.ratecv(
                            chunk,
                            2,
                            1,
                            args.rate,
                            16000,
                            ratecv_state,
                        )

                    # Pass converted audio to loop
                    self._event_loop.call_soon_threadsafe(
                        self._state.audio_queue.put_nowait, chunk
                    )

        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.exception("Unexpected error reading audio")

        self._state.audio_queue.put_nowait(bytes())

    ##########################################
    async def _play_response(self, url: str) -> None:
        """Play response wav file from HA"""

        audio_data = None
        try:
            request = requests.get(url, timeout=(10, 15))
            if request.status_code < 300:
                audio_data = request.content

        except (TimeoutError, ConnectionError) as err:
            _LOGGER.error("Exception: %s", err)

        if not audio_data:
            _LOGGER.error("Failed to get audio file at %s", url)
            return

        audio = simpleaudio.play_buffer(
            audio_data,
            self._state.args.channels,
            self._state.args.width,
            self._state.args.rate,
        )
        audio.wait_done()


##########################################
def get_porcupine(state: State) -> Porcupine:
    """Listen for wake word and send audio to Home-Assistant"""

    args = state.args
    if args.keyword_paths is None:
        if args.keywords is None:
            raise ValueError("Either `--keywords` or `--keyword_paths` must be set.")

        keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in args.keywords]

    else:
        keyword_paths = args.keyword_paths

    if args.sensitivities is None:
        args.sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(args.sensitivities):
        raise ValueError(
            "Number of keywords does not match the number of sensitivities."
        )

    try:
        porcupine = pvporcupine.create(
            access_key=args.access_key,
            library_path=args.library_path,
            model_path=args.model_path,
            keyword_paths=keyword_paths,
            sensitivities=args.sensitivities,
        )

    except pvporcupine.PorcupineInvalidArgumentError as err:
        _LOGGER.error(
            "One or more arguments provided to Porcupine is invalid: %s", args
        )
        _LOGGER.error(err)
        raise

    except pvporcupine.PorcupineActivationError as err:
        _LOGGER.error("AccessKey activation error. %s", err)
        raise

    except pvporcupine.PorcupineActivationLimitError:
        _LOGGER.error(
            "AccessKey '%s' has reached it's temporary device limit", args.access_key
        )
        raise

    except pvporcupine.PorcupineActivationRefusedError:
        _LOGGER.error("AccessKey '%s' refused", args.access_key)
        raise

    except pvporcupine.PorcupineActivationThrottledError:
        _LOGGER.error("AccessKey '%s' has been throttled", args.access_key)
        raise

    except pvporcupine.PorcupineError:
        _LOGGER.error("Failed to initialize Porcupine")
        raise

    _LOGGER.info("Porcupine version: %s", porcupine.version)

    keywords = list()
    for item in keyword_paths:
        keyword_phrase_part = os.path.basename(item).replace(".ppn", "").split("_")
        if len(keyword_phrase_part) > 6:
            keywords.append(" ".join(keyword_phrase_part[0:-6]))
        else:
            keywords.append(keyword_phrase_part[0])

    _LOGGER.debug("keywords: %s", keywords)

    return porcupine


##########################################
if __name__ == "__main__":
    args = get_cli_args()
    _LOGGER.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    if args.debug:
        log_format = "[%(filename)12s: %(funcName)18s()] %(levelname)5s %(message)s"
    else:
        log_format = "%(asctime)s %(levelname)5s %(message)s"

    log_stream = logging.StreamHandler(sys.stdout)
    log_stream.setFormatter(logging.Formatter(log_format))

    _LOGGER.addHandler(log_stream)
    _LOGGER.debug(args)

    audio_pipeline = PorcupinePipeline(args)
    with suppress(KeyboardInterrupt):
        audio_pipeline.start()
