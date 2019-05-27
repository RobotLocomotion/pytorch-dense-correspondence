#!/bin/bash
set -exu
root_dir=$(pwd)

# copied from director/distro/travis/install_deps.sh
install_ubuntu_deps_common()
{
  apt-get update
  apt-get install -y \
    build-essential \
    cmake \
    git \
    libglib2.0-dev \
    libqt4-dev \
    libx11-dev \
    libxext-dev \
    libxt-dev \
    lsb-release \
    python-coverage \
    python-dev \
    python-lxml \
    python-numpy \
    python-scipy \
    python-yaml \
    wget \
    xvfb \
    curl

  # if [ "$MAKE_DOCS" = "ON" ]; then
  #   apt-get install -y \
  #     doxygen \
  #     graphviz \
  #     python-pip \
  #     python-sphinx
  #   pip install sphinx_rtd_theme
  # fi

  # if [ "$MAKE_PACKAGE" = "ON" ]; then
  #   apt-get install -y \
  #     curl
  # fi

}

install_ubuntu_deps()
{
  install_ubuntu_deps_common
  apt-get install -y libgl1-mesa-dev
}

# extra things that are needed
apt-get update
apt-get install -y \
  libqt4-opengl-dev

install_ubuntu_deps
