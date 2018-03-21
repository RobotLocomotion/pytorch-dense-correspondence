#!/usr/bin/env directorPython

import os
import argparse

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.constants import *

"""
Runs change detection to compute masks for each image
"""



def run(data_folder, config_file=CHANGE_DETECTION_CONFIG_FILE):
    """
    Runs the change detection pipeline
    :param data_dir:
    :param config_file:
    :return:
    """

    config = utils.getDictFromYamlFilename(config_file)
    changeDetection, obj_dict = change_detection.ChangeDetection.from_data_folder(data_folder, config=config)
    changeDetection.run()

    app = obj_dict['app']
    app.app.quit()


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

    run(data_folder, config_file=args.config_file)
