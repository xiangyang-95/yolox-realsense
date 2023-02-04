#!/bin/bash

HOME_DIR=$PWD

echo "Installing dependencies ..."
sudo apt update && \
    sudo apt install -y python3-pip

echo "Installing necessary python packages ..."
python3 -m pip install -r requirements.txt

echo "Creating model directory"
mkdir -p model

echo "Downloading YOLOX model"
cd $PWD/model
omz_downloader --name yolox-tiny
omz_converter --name yolox-tiny
