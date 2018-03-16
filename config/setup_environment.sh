export DC_SOURCE_DIR=~/code

use_director(){
    export PATH=$PATH:~/director/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATHPATH:~/director/lib
}

export -f use_director
