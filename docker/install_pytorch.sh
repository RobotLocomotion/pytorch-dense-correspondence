#!/bin/bash

set -euxo pipefail
# pytorch 0.3, CUDA 9
# sudo pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
# sudo pip install torchvision==0.2.1


# pytorch 1.0, CUDA 10
pip install numpy torchvision_nightly
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html


#sudo apt-get install python3-pip
#sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
#sudo pip3 install torchvision

