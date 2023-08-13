#!/bin/bash

echo "Warning: Using picovoice in a docker container"
echo "         will result in consuming a user device"
echo "         for each container that is run. This"
echo "         will exhaust a free user's monthly limit"

docker build -f Dockerfile -t porcupine-pipeline .
