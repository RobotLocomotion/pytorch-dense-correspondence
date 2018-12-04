#!/bin/bash

set -euxo pipefail

sudo apt-get update
sudo pip install requests
sudo pip install matplotlib
sudo pip install scipy
sudo pip install imageio

sudo pip install scikit-image

sudo pip install tensorboard_logger \
    tensorflow

# seems that we need this version of tensorboard
# maybe because tensorboard_logger is not compatible 
# with newer versions of tensorboard?
sudo pip install tensorboard==1.8.0

sudo pip install sklearn

sudo pip install opencv-contrib-python


sudo apt install python-tk \
    ffmpeg

# The dependency for poser
sudo apt install libpcl-dev
sudo apt install libopencv-dev
sudo apt install libproj-dev
sudo apt install libgoogle-glog-dev

# The google test need to build
sudo apt install libgtest-dev
sudo mkdir /usr/src/gtest/build && cmake /usr/src/gtest/ -B/usr/src/gtest/build && sudo make --directory=/usr/src/gtest/build && sudo cp /usr/src/gtest/build/*.a /usr/lib