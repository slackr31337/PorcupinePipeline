"""Function to parse CLI arguments"""

import argparse


def cli_args(args: list = None, keywords: list = None):
    """Parse CLI arguments and return argparse namespace"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--access_key",
        help="AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)",
        required=True,
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        help=f"List of default keywords for detection. Available keywords: {keywords}",
        choices=keywords,
        metavar="",
    )

    parser.add_argument(
        "--keyword_paths",
        nargs="+",
        help=(
            "Absolute paths to keyword model files. "
            "If not set it will be populated from `--keywords` argument"
        ),
    )

    parser.add_argument(
        "--library_path",
        help=(
            "Absolute path to dynamic library. "
            "Default: using the library provided by `pvporcupine`"
        ),
    )

    parser.add_argument(
        "--model_path",
        help=(
            "Absolute path to the file containing model parameters. "
            "Default: using the library provided by `pvporcupine`"
        ),
    )

    parser.add_argument(
        "--sensitivities",
        nargs="+",
        help=(
            "Sensitivities for detecting keywords. Each value should be a number within [0, 1]. "
            "A higher sensitivity results in fewer misses at the cost of increasing "
            "the false alarm rate. If not set 0.5 will be used."
        ),
        type=float,
        default=None,
    )

    parser.add_argument(
        "--audio_device_index",
        help="Index of input audio device.",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--output_path",
        help="Absolute path to recorded audio for debugging.",
        default=None,
    )

    parser.add_argument("--show_audio_devices", action="store_true")

    args = parser.parse_args()
    return args
