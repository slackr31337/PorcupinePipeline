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
import threading
from dataclasses import dataclass, field
from typing import Optional
import time
from datetime import datetime


import aiohttp
import pvporcupine
from pvrecorder import PvRecorder
from playsound import playsound


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="Rate of input audio (hertz)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2,
        help="Width of input audio samples (bytes)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of input audio channels",
    )
    parser.add_argument(
        "--samples-per-chunk",
        dest="samples_per_chunk",
        type=int,
        default=1024,
        help="Number of samples to read at a time from stdin",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TOKEN", "missing_token"),
        help="Home-assistant authentication token",
    )
    parser.add_argument(
        "--pipeline", help="Name of HA pipeline to use (default: preferred)"
    )
    parser.add_argument(
        "--follow-up",
        dest="follow_up",
        action="store_true",
        help="Keep pipeline open after keyword for follow up",
    )
    parser.set_defaults(follow_up=False)
    parser.add_argument(
        "--server",
        default=os.environ.get("SERVER", "localhost"),
        help="host of Home-assistant server",
    )
    parser.add_argument(
        "--server-port",
        dest="server_port",
        type=int,
        default=os.environ.get("SERVER_PORT", 8123),
        help="port of Home-assistant server",
    )
    parser.add_argument(
        "--server-https",
        action="store_true",
        default=bool(os.environ.get("SERVER_HTTPS")),
        help="Use https to connect to Home-assistant server",
    )
    parser.add_argument(
        "--access-key",
        dest="access_key",
        default=os.environ.get("ACCESS_KEY", "missing_access_key"),
        help="AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help=(
            "List of default keywords for detection. "
            f"Available keywords: {sorted(pvporcupine.KEYWORDS)}"
        ),
        default=list(os.environ.get("KEYWORDS", ["porcupine", "computer"])),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar="",
    )
    parser.add_argument(
        "--keyword-paths",
        dest="keyword_paths",
        nargs="+",
        help=(
            "Absolute paths to keyword model files. If not set "
            "it will be populated from `--keywords` argument"
        ),
    )
    parser.add_argument(
        "--library-path",
        dest="library_path",
        help=(
            "Absolute path to dynamic library. "
            "Default: using the library provided by `pvporcupine`"
        ),
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="Absolute path to the file containing model parameters. "
        "Default: using the library provided by `pvporcupine`",
    )
    parser.add_argument(
        "--sensitivities",
        nargs="+",
        help=(
            "Sensitivities for detecting keywords. Each value should be a number "
            "within [0, 1]. A higher sensitivity results in fewer misses at the cost "
            "of increasing the false alarm rate. If not set 0.5 will be used."
        ),
        type=float,
        default=None,
    )
    parser.add_argument(
        "-adev",
        "--audio-device",
        dest="audio_device",
        help="Index of input audio device. (Default: use default audio device)",
        type=int,
        default=os.environ.get("AUDIO_DEVICE", -1),
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        help="Absolute path to recorded audio for debugging.",
        default=None,
    )
    parser.add_argument(
        "--show-audio-devices",
        dest="show_audio_devices",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )

    args = parser.parse_args()
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

    _LOGGER.info("Starting audio pipline for voice assistant")
    state = State(args=args)
    porcupine = get_porcupine(state)
    if not porcupine:
        return

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
    url = f"ws://{args.server}:{args.server_port}/api/websocket"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as websocket:
            _LOGGER.info("Authenticating: %s", url)

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
                while not state.recording:
                    time.sleep(0.1)

                # Run pipeline
                _LOGGER.debug("Starting pipeline")

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
                            _LOGGER.info(event_data.get("message"))
                            state.recording = False

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
        # bytes_per_chunk = args.samples_per_chunk * args.width * args.channels
        rate = args.rate
        width = args.width
        channels = args.channels
        ratecv_state = None

        _LOGGER.debug("Reading audio")
        recorder = PvRecorder(
            device_index=args.audio_device, frame_length=porcupine.frame_length
        )

        recorder.start()
        state.recording = False

        while state.running:
            pcm = recorder.read()
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
                if channels != 1:
                    chunk = audioop.tomono(chunk, width, 1.0, 1.0)

                if width != 2:
                    chunk = audioop.lin2lin(chunk, width, 2)

                if rate != 16000:
                    chunk, ratecv_state = audioop.ratecv(
                        chunk,
                        2,
                        1,
                        rate,
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
