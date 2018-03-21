#!/usr/bin/env directorPython

import os
import argparse

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
import dense_correspondence_manipulation.utils.utils as utils

"""
Runs change detection to compute masks for each image
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/home/manuelli/code/data_volume/sandbox/drill_scenes/01_drill')

    default_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection.yaml')
    parser.add_argument("--config_file", type=str, default=default_config_file)

    parser.add_argument('--current_dir', action='store_true', default=False, help="run the script with --data_dir set to the current directory")



    args = parser.parse_args()
    data_folder = args.data_dir

    if args.current_dir:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()

    config = utils.getDictFromYamlFilename(args.config_file)

    obj_dict = change_detection.setupChangeDetection(data_folder, config=config)
    changeDetection = obj_dict['changeDetection']
    changeDetection.run()
