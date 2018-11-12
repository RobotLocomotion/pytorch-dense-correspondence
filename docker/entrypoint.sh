#!/bin/bash
set -e

export DATA_DIR=~/data
export DC_DATA_DIR=$DATA_DIR/pdc
export DC_SOURCE_DIR=~/code


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
