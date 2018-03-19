export DC_SOURCE_DIR=~/code

export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/modules

use_director(){
    export PATH=$PATH:~/director/bin
}

export -f use_director
