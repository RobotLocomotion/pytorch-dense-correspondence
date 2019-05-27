#!/bin/bash
#
# This script is run by the dockerfile during the docker build.
#

# don't use -u option due to https://github.com/pypa/virtualenv/issues/150
set -ex

root_dir=$USER_HOME_DIR

apt-get update
apt install python3-venv
apt install python3-dev
apt install python3-tk

virtual_env_folder=$PYTHON3_PYTORCH_VIRTUALENV_DIR

python3 -m venv $virtual_env_folder # create virtualenv
source $virtual_env_folder/bin/activate # activate virtualenv

venv_pip=$virtual_env_folder/bin/pip
venv_python=$virtual_env_folder/bin/python


# it seems these are needed following the suggestions here
# https://stackoverflow.com/questions/48561981/activate-python-virtualenv-in-dockerfile?rq=1
pip install \
	ipython \
	jupyter \

pip install --upgrade setuptools
pip install --upgrade pip



# install pytorch 1.0 nightly
pip install numpy torchvision_nightly
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html


# maskrcnn_benchmark and coco api dependencies
pip install \
	ninja \
	yacs \
	cython \
	matplotlib \
    requests \
    opencv-python \
	open3d-python \
    attrs \
    pyyaml

# create directory for stashing some dependencies
cd $root_dir
tmp_dir=$root_dir/github_python3
mkdir $tmp_dir

# # install torchvision
# cd $tmp_dir
# git clone https://github.com/pytorch/vision.git
# cd vision
# python setup.py install

# install pycocotools
cd $tmp_dir
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install


install_maskrcnn_benchmark(){
	# install PyTorch Detection
	# do this once you launch the container
	cd $root_dir/github_venv
	git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
	cd maskrcnn-benchmark
	git checkout 5f2a8263a1a0f2f5f0137042cd4ba64efcb6859a
	# the following will install the lib with
	# symbolic links, so that you can modify
	# the files if you want and won't need to
	# re-build it
	python setup.py build develop	
}

deactivate

