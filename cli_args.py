"""Parse CLI arguments"""

import os
import argparse


import pvporcupine


##########################################
def get_cli_args() -> argparse.Namespace:
    """Parse CLI arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        default=os.environ.get("TOKEN", "missing_token"),
        help="Home-Assistant authentication token",
    )
    parser.add_argument(
        "--pipeline",
        default=os.environ.get("PIPELINE"),
        help="Name of Home-Assistant voice assistant pipeline to use (default: preferred)",
    )
    parser.add_argument(
        "--follow-up",
        dest="follow_up",
        action="store_true",
        default=bool(os.environ.get("FOLLOW_UP")),
        help="Keep pipeline open after keyword for follow up",
    )
    parser.set_defaults(follow_up=False)
    parser.add_argument(
        "--server",
        default=os.environ.get("SERVER", "localhost"),
        help="Hostname or IP address of Home-Assistant serve",
    )
    parser.add_argument(
        "--server-port",
        dest="server_port",
        type=int,
        default=os.environ.get("SERVER_PORT", 8123),
        help="TCP port of Home-Assistant server",
    )
    parser.add_argument(
        "--server-https",
        action="store_true",
        default=bool(os.environ.get("SERVER_HTTPS")),
        help="Use https to connect to Home-Assistant server",
    )
    parser.add_argument(
        "--access-key",
        dest="access_key",
        default=os.environ.get("ACCESS_KEY", "missing_access_key"),
        help="Access Key obtained from Picovoice Console (https://console.picovoice.ai/)",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help=(
            "List of default wakewords for detection. "
            f"Available keywords: {sorted(pvporcupine.KEYWORDS)}"
        ),
        default=list(os.environ.get("KEYWORDS", list(pvporcupine.KEYWORDS))),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar="",
    )
    parser.add_argument(
        "--keyword-paths",
        dest="keyword_paths",
        nargs="+",
        default=os.environ.get("KEYWORD_PATHS"),
        help=(
            "Absolute paths to keyword model files. If not set "
            "it will be populated from `--keywords` argument"
        ),
    )
    parser.add_argument(
        "--library-path",
        dest="library_path",
        default=os.environ.get("LIBRARY_PATH"),
        help=(
            "Absolute path to dynamic library. "
            "Default: using the library provided by `pvporcupine`"
        ),
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=os.environ.get("MODEL_PATH"),
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
        "--dev",
        "--audio-device",
        dest="audio_device",
        help="Index number of input audio device. (Default: use system default audio device)",
        type=int,
        default=os.environ.get("AUDIO_DEVICE", 1),
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=os.environ.get("AUDIO_RATE", 16000),
        help="Rate of input audio (hertz)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=os.environ.get("AUDIO_WIDTH", 2),
        help="Width of input audio samples (bytes)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=os.environ.get("AUDIO_CHANNELS", 1),
        help="Number of input audio channels",
    )
    parser.add_argument(
        "--samples-per-chunk",
        dest="samples_per_chunk",
        type=int,
        default=os.environ.get("AUDIO_SAMPLES_PER_CHUNK", 1024),
        help="Number of audio samples to read at a time",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        default=os.environ.get("OUTPUT_PATH"),
        help="Absolute path to recorded audio for debugging.",
    )
    parser.add_argument(
        "--show-audio-devices",
        "-L",
        dest="show_audio_devices",
        help="Print available devices on system to record audio and exit",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )

    return parser.parse_args()
