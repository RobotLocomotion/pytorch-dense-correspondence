#!/bin/bash

set -euxo pipefail

sudo pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
sudo pip install torchvision

sudo apt-get install python3-pip
sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
sudo pip3 install torchvision