#!/bin/bash

set -euxo pipefail

sudo apt-get update
sudo pip install requests
sudo pip install matplotlib
sudo pip install scipy
sudo pip install imageio

sudo pip install scikit-image
sudo pip install tensorboard

sudo pip install sklearn

sudo pip install opencv-contrib-python

sudo pip install tensorboard_logger \
    tensorflow

sudo apt install python-tk