#!/bin/bash

set -euxo pipefail

apt-get update
apt install --no-install-recommends \
  terminator \
  tmux \
  vim \
  gedit \
  git \
  openssh-client \
  unzip \
  htop \
  libopenni-dev \
  apt-utils \
  usbutils \
  dialog \
  python-pip \
  python-dev \
  ffmpeg \
  nvidia-settings \
  cmake-curses-gui \
  libyaml-dev
apt-get install -y python-setuptools
#pip install --upgrade pip==9.0.3
#apt-get install python-setuptools
#pip install --upgrade pip
pip install --upgrade pip==20.3.4
pip install -U setuptools



pip install \
  jupyter \
  opencv-python==4.2.0.32 \
  plyfile \
  pandas \
  tensorflow \
  future \
  typing

apt-get -y install ipython ipython-notebook
