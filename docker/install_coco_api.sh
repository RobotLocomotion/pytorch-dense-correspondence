#!/bin/bash
#
# This script is run by the dockerfile during the docker build.
#

set -exu

root_dir=$(pwd)

# maskrcnn_benchmark and coco api dependencies
pip install \
    ninja \
    yacs \
    cython \
    matplotlib

mkdir -p $root_dir/github
cd $root_dir/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install


cd $root_dir