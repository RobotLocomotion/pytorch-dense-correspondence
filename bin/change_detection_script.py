#!/usr/bin/env directorPython

import os
import shutil
import time

import dense_correspondence_manipulation.scripts.run_change_detection_pipeline as run_change_detection_pipeline
from dense_correspondence_manipulation.utils.constants import *


"""
Script that runs the change detection pipeline on all subfolders in a directory

Usage:

    change_detection_script.py 

# Make sure to use appropriate calls to setup environment, and call this script from a directory with many subdirectories of scenes
# For example:

    cd pytorch-dense-correspondence/data_volume
    use_pytorch_dense_correspondence
    use_director
    change_detection_script.py

"""

def main():

    parent_folder = os.getcwd()
    list_of_dirs = os.listdir(parent_folder)
    print "list_of_dirs", list_of_dirs

    num_dirs = len(list_of_dirs)
    # list_of_dirs = ["04_drill_long_downsampled", "05_drill_long_downsampled"]

    for idx, dir in enumerate(list_of_dirs):

        data_folder = os.path.join(parent_folder, dir)

        image_masks_folder = os.path.join(data_folder, 'image_masks')
        if os.path.isdir(image_masks_folder):
            shutil.rmtree(image_masks_folder)

        print "Processing scene %d of %d" %(idx, num_dirs)
        print "Running change detection for %s" %(dir)

        cmd = "run_change_detection_pipeline.py --data_dir " + data_folder
        print "cmd: ", cmd
        os.system(cmd)

        # this doesn't work, for some reason different directorApp's won't quit.
        # run_change_detection_pipeline.run(data_folder, CHANGE_DETECTION_CONFIG_FILE)

        time.sleep(2.0)

if __name__== "__main__":
    main()