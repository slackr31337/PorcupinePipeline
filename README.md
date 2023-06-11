# PorcupinePipeline

Use a raspberry pi and microphone to trigger a wake word
for a Home-Assistant Voice Assistant pipeline conversation

<https://www.home-assistant.io/voice_control/>

## Usage

    run_pipeline.py
    [-h] [--token TOKEN] [--pipeline PIPELINE] [--follow-up] [--server SERVER] [--server-port SERVER_PORT] [--server-https]
    [--access-key ACCESS_KEY] [--keywords  [...]] [--keyword-paths KEYWORD_PATHS [KEYWORD_PATHS ...]] [--library-path LIBRARY_PATH]
    [--model-path MODEL_PATH] [--sensitivities SENSITIVITIES [SENSITIVITIES ...]] [--dev AUDIO_DEVICE] [--rate RATE] [--width WIDTH] [--channels CHANNELS] [--samples-per-chunk SAMPLES_PER_CHUNK [--output-path OUTPUT_PATH] [--show-audio-devices] [-d]

optional arguments:

    -h, --help
        Show this help message and exit

    --token TOKEN
        Home-Assistant authentication token

    --pipeline PIPELINE
        Name of Home-Assistant voice assistant to use (default: preferred)

    --follow-up
        Keep pipeline open after keyword for follow up

    --server SERVER
        Hostname or IP address of Home-Assistant server

    --server-port SERVER_PORT
        TCP port of Home-Assistant server

    --server-https
        Use https to connect to Home-Assistant server

    --access-key ACCESS_KEY
        Access Key obtained from Picovoice Console (https://console.picovoice.ai/)

    --keywords []
        List of default keywords for detection.
        Available keywords: ['alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 'grapefruit', 'grasshopper', 'hey barista', 'hey google', 'hey siri', 'jarvis', 'ok google', 'pico clock', 'picovoice', 'porcupine', 'smart mirror', 'snowboy', 'terminator', 'view glass']

    --keyword-paths KEYWORD_PATHS [KEYWORD_PATHS ...]
        Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument
    
    --library-path LIBRARY_PATH
        Absolute path to dynamic library. Default: using the library provided by `pvporcupine`

    --model-path MODEL_PATH
        Absolute path to the file containing model parameters. Default: using the library provided by `pvporcupine`

    --sensitivities SENSITIVITIES [SENSITIVITIES ...]
        Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 will be used.

    --audio-device AUDIO_DEVICE
        Index number of input audio device. (Default: use system default audio device)

    --output-path OUTPUT_PATH
        Absolute path to recorded audio for debugging.

    --show-audio-devices
        Print available devices on system to record audio and exit

    --debug
        Print DEBUG messages to console

## Example

Required Porcupine access_key and Home-Assistant token

    - From Picovoice access_key for Porcupine wake word detection
    - From Home-Assistant long-lived access token

Run:

    cd /usr/src
    git clone https://github.com/slackr31337/PorcupinePipeline.git

    export ACCESS_KEY='my-picovoice-access-key'
    export TOKEN='my-home-assistant-token'
    export SERVER=192.168.0.10
    python3 ./run_pipeline.py --server `home-assistant-host` -pipeline 'OpenAI' --follow-up

Docker:

    cd /usr/src
    git clone https://github.com/slackr31337/PorcupinePipeline.git

    cd PorcupinePipeline
    bash build.sh

    export ACCESS_KEY='my-picovoice-access-key'
    export TOKEN='my-home-assistant-token'

    docker run --rm \
        --env SERVER=192.168.0.10 \
        --env SERVER_PORT=8123 \
        --env AUDIO_DEVICE=3 \
        --env ACCESS_KEY=${ACCESS_KEY} \
        --ENV TOKEN=${TOKEN} \
        --device /dev/snd
        porcupine-pipeline

Environment Variables:



## Used code from the following projects

    - https://github.com/synesthesiam/homeassistant-pipeline/blob/master/audio_to_audio.py

    - https://github.com/Picovoice/porcupine/blob/master/demo/python/porcupine_demo_mic.py

    https://github.com/Picovoice/porcupine
    AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)
