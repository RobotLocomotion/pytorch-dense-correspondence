#!/bin/bash
#
# This script is run by the dockerfile during the docker build.
#

set -exu

root_dir=$USER_HOME_DIR
install_dir=$root_dir/install

apt-get update

# convenience programs to have inside the docker
apt install --no-install-recommends \
  wget \
  libglib2.0-dev \
  libqt4-dev \
  libqt4-opengl \
  libx11-dev \
  libxext-dev \
  libxt-dev \
  mesa-utils \
  libglu1-mesa-dev \
  python-dev \
  python-lxml \
  python-numpy \
  python-scipy \
  python-yaml

build_director()
{
  director_version=0.1.0-318-gd10dfa9
  director_binary=ubuntu-16.04
  wget https://dl.bintray.com/patmarion/director/director-$director_version-$director_binary.tar.gz

  tar -xzf director-$director_version-$director_binary.tar.gz
  mv director-$director_version-$director_binary director
  rm -rf director-$director_version-$director_binary.tar.gz
}

build_director

