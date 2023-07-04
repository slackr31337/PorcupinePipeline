# PorcupinePipeline

Use a raspberry pi and microphone to trigger a wake word
for a conversation with a Home-Assistant Voice Assistant

<https://www.home-assistant.io/voice_control/>

## Usage and Example

Required Authentication

    - From Picovoice access_key for Porcupine wake word detection
    - From Home-Assistant long-lived access token

Environment Variables:

    SERVER              Home-Assistant FQDN
    SERVER_PORT         Home-Assistant port
    SERVER_HTTPS        Home-Assistant using https
    TOKEN               Home-Assistant long-lived access token
    PIPELINE            Home-Assistant voice pipeline name

    ACCESS_KEY          Picovoice access_key
    KEYWORDS            Wake words to trigger voice pipeline
    KEYWORD_PATHS       Absolute paths to keyword model files
    LIBRARY_PATH        Absolute path to dynamic library
    MODEL_PATH          Absolute path to the file containing model parameters

    AUDIO_DEVICE        Index of audio device for microphone
    AUDIO_RATE          Rate of input audio
    AUDIO_WIDTH         Width of input audio sample
    AUDIO_CHANNELS      Number of input audio channels
    OUTPUT_PATH         Absolute path to recorded audio

Run:

    cd /usr/src
    git clone https://github.com/slackr31337/PorcupinePipeline.git

    export ACCESS_KEY='my-picovoice-access-key'
    export TOKEN='my-home-assistant-token'
    export SERVER=home-assistant.local
    export PIPELINE=OpenAI
    export AUDIO_DEVICE=1

    python3 ./voice_pipeline.py --server home-assistant.local --pipeline 'OpenAI' --follow-up

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
        --env TOKEN=${TOKEN} \
        --device /dev/snd
        porcupine-pipeline

Tested with:

    wyoming-piper <https://hub.docker.com/r/rhasspy/wyoming-piper>
    wyoming-whisper <https://hub.docker.com/r/rhasspy/wyoming-whisper>
    Home-Assistant 2023.06.1 <https://www.home-assistant.io/>

## Used code from the following projects

    - https://github.com/synesthesiam/homeassistant-pipeline/blob/master/audio_to_audio.py

    - https://github.com/Picovoice/porcupine/blob/master/demo/python/porcupine_demo_mic.py

    https://github.com/Picovoice/porcupine
    AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)
