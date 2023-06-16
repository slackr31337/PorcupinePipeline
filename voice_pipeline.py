"""
Run Porcupine wake word listener 
and send audio to Home Assistant audio pipeline"""
from __future__ import annotations

import os
import sys
import struct
import logging
import argparse
import asyncio
import audioop
import ssl
import threading
from dataclasses import dataclass, field
from typing import Optional
import time
from datetime import datetime
import warnings


import aiohttp
import pvporcupine
from pvrecorder import PvRecorder
from playsound import playsound
from cli_args import get_cli_args

warnings.filterwarnings("ignore", category=DeprecationWarning)
_LOGGER = logging.getLogger(__name__)


##########################################
@dataclass
class State:
    """Client state."""

    args: argparse.Namespace
    running: bool = True
    recording: bool = False
    audio_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)


##########################################
async def main() -> None:
    """Main entry point."""

    args = get_cli_args()
    proto = "http"
    if args.server_https:
        proto += "s"

    args.ha_url = f"{proto}://{args.server}:{args.server_port}"

    _LOGGER.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    if args.debug:
        log_format = "[%(filename)12s: %(funcName)18s()] %(levelname)5s %(message)s"
    else:
        log_format = "%(levelname)5s %(message)s"

    log_stream = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format)
    log_stream.setFormatter(formatter)
    _LOGGER.addHandler(log_stream)
    _LOGGER.debug(args)

    _LOGGER.info("Starting Porcupine listener")
    state = State(args=args)
    porcupine = get_porcupine(state)
    if not porcupine:
        return

    _LOGGER.info("Starting audio pipline thread")
    audio_thread = threading.Thread(
        target=read_audio,
        args=(state, asyncio.get_running_loop(), porcupine),
        daemon=True,
    )
    audio_thread.start()

    try:
        await loop_pipeline(state)

    except KeyboardInterrupt:
        pass

    finally:
        state.recording = False
        state.running = False
        audio_thread.join()


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
async def loop_pipeline(state: State) -> None:
    """Run pipeline in a loop, executing voice commands and printing TTS URLs."""

    args = state.args
    url = "ws"

    sslcontext = None
    if args.server_https:
        sslcontext = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
        url = "https"

    url += f"://{args.server}:{args.server_port}/api/websocket"

    async with aiohttp.ClientSession() as session:
        _LOGGER.info("Authenticating: %s", url)

        async with session.ws_connect(url, ssl=sslcontext) as websocket:
            msg = await websocket.receive_json()
            assert msg["type"] == "auth_required", msg

            await websocket.send_json(
                {
                    "type": "auth",
                    "access_token": args.token,
                }
            )

            msg = await websocket.receive_json()
            _LOGGER.debug(msg)
            assert msg["type"] == "auth_ok", msg

            _LOGGER.info("Authenticated with Home Assistant successfully")

            message_id = 1
            pipeline_id: Optional[str] = None

            if args.pipeline:
                _LOGGER.info("Using Home-Assistant pipeline %s", args.pipeline)

                # Get list of available pipelines and resolve name
                await websocket.send_json(
                    {
                        "type": "assist_pipeline/pipeline/list",
                        "id": message_id,
                    }
                )
                msg = await websocket.receive_json()
                _LOGGER.debug(msg)
                message_id += 1

                pipelines = msg["result"]["pipelines"]
                for pipeline in pipelines:
                    if pipeline["name"] == args.pipeline:
                        pipeline_id = pipeline["id"]
                        break

                if not pipeline_id:
                    raise ValueError(
                        f"No pipeline named {args.pipeline} in {pipelines}"
                    )

            # Pipeline loop
            _LOGGER.info("Starting audio processing loop")
            while state.running:
                # Clear audio queue
                while not state.audio_queue.empty():
                    state.audio_queue.get_nowait()

                count = 0
                _LOGGER.info(
                    "[%s] Waiting for wake word to trigger audio", datetime.now()
                )
                while not state.recording:
                    time.sleep(0.2)

                # Run pipeline
                _LOGGER.info(
                    "[%s] Listening and sending audio to voice pipeline", datetime.now()
                )

                pipeline_args = {
                    "type": "assist_pipeline/run",
                    "id": message_id,
                    "start_stage": "stt",
                    "end_stage": "tts",
                    "input": {
                        "sample_rate": 16000,
                    },
                }
                if pipeline_id:
                    pipeline_args["pipeline"] = pipeline_id

                await websocket.send_json(pipeline_args)
                message_id += 1

                msg = await websocket.receive_json()
                _LOGGER.debug(msg)

                assert msg["success"], "Pipeline failed to run"

                # Get handler id.
                # This is a single byte prefix that needs to be in every binary payload.
                msg = await websocket.receive_json()
                _LOGGER.debug(msg)

                handler_id = bytes(
                    [msg["event"]["data"]["runner_data"]["stt_binary_handler_id"]]
                )

                # Audio loop for single pipeline run
                receive_event_task = asyncio.create_task(websocket.receive_json())
                while True:
                    audio_chunk = await state.audio_queue.get()

                    # Prefix binary message with handler id
                    send_audio_task = asyncio.create_task(
                        websocket.send_bytes(handler_id + audio_chunk)
                    )
                    pending = {send_audio_task, receive_event_task}
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if receive_event_task in done:
                        event = receive_event_task.result()
                        _LOGGER.debug(event)
                        event_type = event["event"]["type"]

                        if event_type == "run-end":
                            count += 1
                            _LOGGER.debug("[%s] Pipeline finished", count)
                            if args.follow_up and count < 4:
                                state.recording = True
                            else:
                                state.recording = False
                            break

                        event_data = event["event"].get("data")
                        if event_type == "error":
                            state.recording = False
                            _LOGGER.info(
                                "[%s] %s. Listening stopped",
                                datetime.now(),
                                event_data.get("message"),
                            )
                            break

                        elif event_type == "stt-end":
                            speech = event_data["stt_output"].get("text")
                            _LOGGER.info(
                                "[%s] Recongized speech: %s", datetime.now(), speech
                            )

                        elif event_type == "tts-end":
                            # URL of text to speech audio response (relative to server)
                            tts_url = args.ha_url
                            tts_url += event_data["tts_output"].get("url")
                            _LOGGER.info(
                                "[%s] Play response: %s", datetime.now(), tts_url
                            )
                            playsound(tts_url)

                        receive_event_task = asyncio.create_task(
                            websocket.receive_json()
                        )

                    if send_audio_task not in done:
                        await send_audio_task


##########################################
def read_audio(
    state: State, loop: asyncio.AbstractEventLoop, porcupine: pvporcupine
) -> None:
    """Reads chunks of raw audio from standard input."""
    try:
        args = state.args
        keywords = args.keywords
        ratecv_state = None

        _LOGGER.debug("Reading audio")
        recorder = PvRecorder(
            device_index=args.audio_device, frame_length=porcupine.frame_length
        )

        recorder.start()
        state.recording = False

        while state.running:
            pcm = recorder.read()

            if not state.recording:
                result = porcupine.process(pcm)
                # _LOGGER.debug("porcupine result: %s", result)

                if result >= 0:
                    _LOGGER.info(
                        "[%s] Detected keyword `%s`", datetime.now(), keywords[result]
                    )
                    state.recording = True

            if state.recording:
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
                loop.call_soon_threadsafe(state.audio_queue.put_nowait, chunk)

    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.exception("Unexpected error reading audio")

    state.audio_queue.put_nowait(bytes())


##########################################
if __name__ == "__main__":
    asyncio.run(main())
