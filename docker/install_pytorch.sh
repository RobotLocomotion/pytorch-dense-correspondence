#!/bin/bash

set -euxo pipefail

# pytorch 1.0, CUDA 10
pip install numpy torchvision_nightly
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html


