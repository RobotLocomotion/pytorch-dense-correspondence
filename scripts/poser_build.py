#!/usr/bin/env python
import os
import shutil

if __name__ == "__main__":
    poser_build_dir = os.getenv("POSER_BUILD_DIR")
    if os.path.isdir(poser_build_dir):
        shutil.rmtree(poser_build_dir)


    os.makedirs(poser_build_dir)

    os.chdir(poser_build_dir)
    poser_cmakelists_file = os.path.join(os.getenv("DC_SOURCE_DIR"), 'src', 'poser', 'CMakeLists.txt')
    poser_dir = os.path.join(os.getenv("DC_SOURCE_DIR"), 'src', 'poser')
    cmd = "cmake  -DCMAKE_BUILD_TYPE=Release %s" %(poser_dir)
    print "cmd:", cmd
    os.system(cmd)

    build_cmd = "make -j"
    os.system(build_cmd)
