#!/bin/bash

set -euxo pipefail

# pytorch 1.1, CUDA 10
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp27-cp27mu-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp27-cp27mu-linux_x86_64.whl

# if the above command does not work, then you have python 2.7 UCS2, use this command
# pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp27-cp27m-linux_x86_64.whl
# pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp27-cp27m-linux_x86_64.whl


