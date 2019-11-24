#!/bin/bash
set -e

# assumes that the following environment variables have been set
# typically in entrypoint.sh if inside docker
#  export PYTHON_MODULES_ROOT=$HOME/code
#  export KEY_DYNAM_ROOT=$PYTHON_MODULES_ROOT/key_dynam
#  export DC_SOURCE_DIR=$PYTHON_MODULES_ROOT/pdc


function use_pdc()
{
	export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/modules
  export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR
#  export PATH=$PATH:$DC_SOURCE_DIR/bin
#  export PATH=$PATH:$DC_SOURCE_DIR/modules/dense_correspondence_manipulation/scripts
}

export -f use_pdc

exec "$@"
