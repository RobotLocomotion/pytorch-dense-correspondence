export DC_SOURCE_DIR=~/code

export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/modules
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR
export PATH=$PATH:$DC_SOURCE_DIR/bin
export PATH=$PATH:$DC_SOURCE_DIR/modules/dense_correspondence_manipulation/scripts
use_director(){
    export PATH=$PATH:$DC_SOURCE_DIR/build/director/install/bin
}

export -f use_director
