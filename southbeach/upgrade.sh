#!/usr/bin/env bash

sudo apt-get update
#sudo apt-get upgrade
sudo apt-get install libportaudio2 portaudio19-dev python3-pyaudio
sudo pip3 install -r requirements.txt
