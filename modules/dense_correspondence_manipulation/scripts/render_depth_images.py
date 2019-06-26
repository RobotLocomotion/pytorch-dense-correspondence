#!/usr/bin/env directorPython

import os
import argparse

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.constants import *
from director.timercallback import TimerCallback


"""
Renders depth images against the entire scene
"""

CONFIG_FILE = CHANGE_DETECTION_CONFIG_FILE


def run(data_folder, config_file=CONFIG_FILE, debug=False, globalsDict=None):
    """
    Runs the change detection pipeline
    :param data_dir:
    :param config_file:
    :return:
    """

    if globalsDict is None:
        globalsDict = globals()




    config_file = CONFIG_FILE
    config = utils.getDictFromYamlFilename(config_file)

    # make dimensions large so no cropping
    for key in config['crop_box']['dimensions']:
        config['crop_box']['dimensions'][key] = 10.0# set it to 10 meteres


    changeDetection, obj_dict = change_detection.ChangeDetection.from_data_folder(data_folder, config=config, globalsDict=globalsDict,
                                                                                  background_data_folder=data_folder)

    # set foreground mesh to actually be background mesh
    changeDetection.foreground_reconstruction = changeDetection.background_reconstruction

    app = obj_dict['app']
    globalsDict['cd'] = changeDetection
    view = obj_dict['view']

    # if debug:
    #     changeDetection.background_reconstruction.visualize_reconstruction(view, name='background')

    def single_shot_function():
        changeDetection.render_depth_images()
        app.app.quit()

    if not debug:
        TimerCallback(callback=single_shot_function).singleShot(0)

    app.app.start(restoreWindow=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/home/manuelli/code/data_volume/sandbox/drill_scenes/04_drill_long_downsampled')

    default_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection.yaml')
    parser.add_argument("--config_file", type=str, default=default_config_file)

    parser.add_argument('--current_dir', action='store_true', default=False, help="run the script with --data_dir set to the current directory")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="launch the app in debug mode")


    globalsDict = globals()
    args = parser.parse_args()
    data_folder = args.data_dir

    if args.current_dir:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()

    run(data_folder, config_file=args.config_file, debug=args.debug, globalsDict=globalsDict)
