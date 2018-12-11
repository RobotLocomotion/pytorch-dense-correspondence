#!/bin/bash

set -euxo pipefail

# The dependency for poser
sudo apt install libpcl-dev
sudo apt install libopencv-dev
sudo apt install libyaml-cpp-dev
sudo apt install libproj-dev
sudo apt install libgoogle-glog-dev

# The google test need to build
sudo apt install libgtest-dev
sudo mkdir /usr/src/gtest/build && cmake /usr/src/gtest/ -B/usr/src/gtest/build && sudo make --directory=/usr/src/gtest/build && sudo cp /usr/src/gtest/build/*.a /usr/lib