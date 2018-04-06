#!/usr/bin/env directorPython

import os
import argparse

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.scripts.convert_ply_to_vtp as convert_ply_to_vtp
import dense_correspondence_manipulation.scripts.run_change_detection as run_change_detection
import dense_correspondence_manipulation.scripts.render_depth_images as render_depth_images

"""
Runs change detection to compute masks for each image
"""


def run(data_folder, config_file):

    # if not os.path.isfile(os.path.join(data_folder, 'images.vtp')):
    #     print "converting ply to vtp . . . . "
    #     convert_ply_to_vtp.run(data_folder)
    #     print "finished converting ply to vtp\n\n"


    print "running change detection . . . "
    run_change_detection.run(data_folder, config_file=config_file)
    print "finished running change detection"

    print "rendering depth images"
    render_depth_images.run(data_folder)
    print "finished rendering depth images"

def run_on_all_subfolders(directory, config_file):

    for dir in os.listdir(directory):
        full_dir = os.path.join(directory, dir)

        if not os.path.isdir(full_dir):
            continue

        # print "full_dir", full_dir
        run(full_dir, config_file)




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

    # run(data_folder, config_file=args.config_file)
    run_on_all_subfolders(data_folder, args.config_file)