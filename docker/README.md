
Warning: Using a docker image is not recommended with a free picovoice account.


To run in Docker:

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
        --device /dev/snd \
        porcupine-pipeline
