#!/bin/bash

export VIRTUAL_ENV=/opt/venv

source $VIRTUAL_ENV/bin/activate
$VIRTUAL_ENV/bin/python3 /app/voice_pipeline.py
