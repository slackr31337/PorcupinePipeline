#!/bin/bash

# Testing on Debian 12 and Python 3.11

sudo apt update
sudo apt install -y python3 python3-dev python3-venv


python3 -m venv .env

source ./.env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt


