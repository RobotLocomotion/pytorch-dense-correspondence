#!/usr/bin/env directorPython

import os
import argparse

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.constants import *
from director.timercallback import TimerCallback

"""
Runs change detection to compute masks for each image
"""

CONFIG_FILE = CHANGE_DETECTION_CONFIG_FILE
# CONFIG_FILE = CHANGE_DETECTION_BACKGROUND_SUBTRACTION_CONFIG_FILE

def run(data_folder, config_file=CONFIG_FILE, debug=False, globalsDict=None,
        background_scene_data_folder=None):
    """
    Runs the change detection pipeline
    :param data_dir: The 'processed' subfolder of a top-level log folder
    :param config_file:
    :return:
    """

    if globalsDict is None:
        globalsDict = globals()

    if background_scene_data_folder is None:
        background_scene_data_folder = data_folder




    config_file = CONFIG_FILE
    config = utils.getDictFromYamlFilename(config_file)


    changeDetection, obj_dict = change_detection.ChangeDetection.from_data_folder(data_folder, config=config, globalsDict=globalsDict,
                                                                                  background_data_folder=background_scene_data_folder)

    app = obj_dict['app']
    globalsDict['cd'] = changeDetection
    view = obj_dict['view']

    # if debug:
    #     changeDetection.background_reconstruction.visualize_reconstruction(view, name='background')

    def single_shot_function():
        changeDetection.run()
        app.app.quit()

    if not debug:
        TimerCallback(callback=single_shot_function).singleShot(0)

    app.app.start(restoreWindow=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, help="full path to the processed/ folder of a top level log folder")

    default_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection.yaml')
    parser.add_argument("--config_file", type=str, default=default_config_file)

    parser.add_argument('--current_dir', action='store_true', default=False, help="run the script with --data_dir set to the current directory. You should be in the processed/ subfolder")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="launch the app in debug mode")


    globalsDict = globals()
    args = parser.parse_args()

    if (not args.current_dir) and (not args.data_dir):
        raise ValueError("You must specify either current_dir or data_dir")
    data_folder = args.data_dir

    if args.current_dir:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()
    elif args.data_dir:
        data_folder = args.data_dir

    run(data_folder, config_file=args.config_file, debug=args.debug, globalsDict=globalsDict)
