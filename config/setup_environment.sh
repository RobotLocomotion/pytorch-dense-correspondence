export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/modules
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR
export PATH=$PATH:$DC_SOURCE_DIR/bin
export PATH=$PATH:$DC_SOURCE_DIR/modules/dense_correspondence_manipulation/scripts


build_maskrcnn_benchmark(){
    cd $DC_SOURCE_DIR/build_maskrcnn_benchmark
    python setup.py build develop

}

export -f build_maskrcnn_benchmark