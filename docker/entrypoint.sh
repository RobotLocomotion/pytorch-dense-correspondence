#!/bin/bash
set -e

export DATA_DIR=~/data
export DC_DATA_DIR=$DATA_DIR/pdc
export DC_SOURCE_DIR=~/code
export PDC_BUILD_DIR=$DC_SOURCE_DIR/build
export POSER_BUILD_DIR=$PDC_BUILD_DIR/poser

# location of custom COCO data
export COCO_CUSTOM_DATA_DIR=$DC_DATA_DIR/coco

function use_pytorch_dense_correspondence()
{
    source $DC_SOURCE_DIR/config/setup_environment.sh
    echo "using pdc"
}

export -f use_pytorch_dense_correspondence

use_director(){
    export PATH=$PATH:~/director/bin
}

export -f use_director

exec "$@"

cd $DC_SOURCE_DIR
