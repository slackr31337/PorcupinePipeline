"""
Run Porcupine wake word listener 
and send audio to Home Assistant audio pipeline
"""
from __future__ import annotations

import os
import sys
import time
import struct
import logging
import argparse
import asyncio
import audioop
import ssl
import threading
import requests
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Optional
import warnings

import aiohttp
import pvporcupine
from pvrecorder import PvRecorder
import simpleaudio


from cli_args import get_cli_args

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.raiseExceptions = False

_LOGGER = logging.getLogger(__name__)

RESULT = "result"
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
class PorcupinePipeline:
    """Class used to process audio pipeline using HA websocket"""

    websocket_url = None
    _websocket = None
    _ha_url = None
    _sslcontext = None
    _message_id = 1
    _last_ping = 0
    _followup = False

    ##########################################
    def __init__(self, args: argparse.Namespace):
        """Setup Websocket client and audio pipeline"""

        self._state = State(args=args)
        self._state.running = False

        self._conn = aiohttp.TCPConnector()
        self._event_loop = asyncio.get_event_loop()

        self._porcupine = get_porcupine(self._state)
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
            self._sslcontext = ssl.create_default_context(
                purpose=ssl.Purpose.CLIENT_AUTH
            )
            proto += "s"
            self.websocket_url = proto

        self._ha_url = f"{proto}://{server}:{port}"
        self.websocket_url += f"://{server}:{port}/api/websocket"

    ##########################################
    def start(self) -> None:
        """Start listening for wake word"""

        self._websocket = None
        self._state.running = True

        _LOGGER.info("Starting audio listener thread")
        self._audio_thread.start()

        self._event_loop.run_until_complete(self._start_audio_pipeline())

    ##########################################
    def stop(self) -> None:
        """Stop audio thread and loop"""

        _LOGGER.info("Stopping")

        self._state.recording = False
        self._state.running = False
        self._audio_thread.join(1)
        self._porcupine = None
        self._websocket = None

    ##########################################
    async def _ping(self):
        """Send Ping to HA"""

        if not self._state.running or not self._state.connected:
            return

        now = int(time.time())
        if now - self._last_ping < 30:
            await asyncio.sleep(0.3)
            return

        await self._send_ws({TYPE: "ping"})
        response = await self._websocket.receive_json(timeout=WEBSOCKET_TIMEOUT)

        assert response[TYPE] == "pong", response
        self._last_ping = int(time.time())

    ##########################################
    async def _send_ws(self, message: dict) -> None:
        """Send websocket JSON message and increment message ID"""

        if not self._state.connected:
            _LOGGER.error("WS not connected")
            return

        if not isinstance(message, dict):
            _LOGGER.error("Invalid WS message type")
            return

        message[ID] = self._message_id
        _LOGGER.debug("send_ws() message=%s", message)

        await self._websocket.send_json(message)
        self._message_id += 1

    ##########################################
    async def _start_audio_pipeline(self):
        """Start HA audio pipeline"""

        _LOGGER.info("Starting audio pipeline loop")

        async with aiohttp.ClientSession(connector=self._conn) as session:
            async with session.ws_connect(
                self.websocket_url, ssl=self._sslcontext, timeout=WEBSOCKET_TIMEOUT
            ) as self._websocket:
                await self._auth_ha()
                await self.get_audio_pipeline()
                await self._process_loop()

    ##########################################
    async def _auth_ha(self) -> None:
        """Authenticate websocket connection to HA"""

        _LOGGER.info("Authenticating to: %s", self.websocket_url)

        self._state.connected = False
        msg = await self._websocket.receive_json()
        assert msg[TYPE] == "auth_required", msg

        await self._websocket.send_json(
            {
                TYPE: "auth",
                "access_token": self._state.args.token,
            }
        )

        msg = await self._websocket.receive_json()
        assert msg.get(TYPE) == "auth_ok", msg
        _LOGGER.info(
            "Authenticated to Home Assistant version %s", msg.get("ha_version")
        )
        self._state.connected = True

    ##########################################
    async def get_audio_pipeline(self) -> None:
        """Return ID of audio pipeline"""

        self._pipeline_id: Optional[str] = None
        if self._state.args.pipeline:
            _LOGGER.info(
                "Using Home Assistant audio pipeline %s", self._state.args.pipeline
            )

            # Get list of available pipelines and resolve name
            await self._send_ws(
                {
                    TYPE: "assist_pipeline/pipeline/list",
                }
            )
            msg = await self._websocket.receive_json()
            _LOGGER.debug(msg)
            if RESULT not in msg:
                _LOGGER.error("FAiled to get audio pipeline from HA")
                _LOGGER.error("response=%s", msg)
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
                ID: self._message_id,
                "start_stage": "stt",
                "end_stage": "tts",
                "input": {
                    "sample_rate": 16000,
                },
            }
            if self._pipeline_id:
                pipeline_args["pipeline"] = self._pipeline_id

            # Send audio pipeline args to HA
            await self._send_ws(pipeline_args)
            msg = await self._websocket.receive_json()
            assert msg["success"], "Pipeline failed to start"
            
            _LOGGER.info(
                "Listening and sending audio to voice pipeline %s", self._pipeline_id
            )
            await self.stt_task()

    ##########################################
    async def stt_task(self) -> None:
        """Create task to process speech to text"""

        # Audio loop for single pipeline run
        count = 0

        # Get handler id.
        # This is a single byte prefix that needs to be in every binary payload.
        msg = await self._websocket.receive_json()
        _LOGGER.debug(msg)

        handler_id = bytes(
            [msg[EVENT][DATA]["runner_data"].get("stt_binary_handler_id")]
        )

        receive_event_task = asyncio.create_task(self._websocket.receive_json())
        _LOGGER.debug("New audio task %s", receive_event_task)

        while self._state.connected:
            audio_chunk = await self._state.audio_queue.get()
            if not audio_chunk:
                _LOGGER.error("No audio chunk in queue")

            # Prefix binary message with handler id
            send_audio_task = asyncio.create_task(
                self._websocket.send_bytes(handler_id + audio_chunk)
            )
            pending = {send_audio_task, receive_event_task}
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receive_event_task in done:
                event = receive_event_task.result()
                if EVENT in event:
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
                    _LOGGER.debug("event_data=%s", event_data)

                receive_event_task = asyncio.create_task(self._websocket.receive_json())

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
            recorder = PvRecorder(
                device_index=args.audio_device,
                frame_length=self._porcupine.frame_length,
            )

            recorder.start()
            self._state.recording = False
            while self._state.running:
                try:
                    pcm = recorder.read()

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

        request = requests.get(url, timeout=(10, 30))
        if request.status_code > 299:
            _LOGGER.error("Failed to get audio file at %s", url)
            return

        audio = simpleaudio.play_buffer(
            request.content,
            self._state.args.channels,
            self._state.args.width,
            self._state.args.rate,
        )
        audio.wait_done()


##########################################
def get_porcupine(state: State) -> pvporcupine:
    """Listen for wake word and send audio to Home-Assistant"""

    args = state.args
    devices = {}
    for idx, device in enumerate(PvRecorder.get_audio_devices()):
        devices[idx] = device
        _LOGGER.info("Device %d: %s", idx, device)

    _id = args.audio_device
    audio_device = devices.get(_id)
    if not audio_device:
        _LOGGER.error("Invalid audio device id: %s", _id)
        return None

    _LOGGER.info("Using Device %d: %s", _id, audio_device)
    if args.show_audio_devices:
        sys.exit(0)

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
        return None

    except pvporcupine.PorcupineActivationError as err:
        _LOGGER.error("AccessKey activation error. %s", err)
        return None

    except pvporcupine.PorcupineActivationLimitError:
        _LOGGER.error(
            "AccessKey '%s' has reached it's temporary device limit", args.access_key
        )
        return None

    except pvporcupine.PorcupineActivationRefusedError:
        _LOGGER.error("AccessKey '%s' refused", args.access_key)
        return None

    except pvporcupine.PorcupineActivationThrottledError:
        _LOGGER.error("AccessKey '%s' has been throttled", args.access_key)
        return None

    except pvporcupine.PorcupineError:
        _LOGGER.error("Failed to initialize Porcupine")
        return None

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
def main() -> None:
    """Main entry point."""

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
    audio_pipeline.start()


##########################################
if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()

    sys.exit(0)
