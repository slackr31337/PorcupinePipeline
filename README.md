# PorcupinePipeline

Use a raspberry pi and microphone to trigger a wake word
for a Home-Assistant Voice Assistant pipeline conversation

# AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)



## Usage

python3 ./run_pipeline.py --server `home-assistant-host` -pipeline 'OpenAI' --follow-up


Based off: https://github.com/synesthesiam/homeassistant-pipeline/blob/master/audio_to_audio.py
Uses: https://github.com/Picovoice/porcupine

