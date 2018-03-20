import argparse
import dense_correspondence_manipulation.change_detection.change_detection as change_detection

"""
Runs change detection to compute masks for each image
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/home/manuelli/code/data_volume/sandbox/drill_scenes/01_drill')

    args = parser.parse_args()
    data_folder = args.data_dir

    obj_dict = change_detection.setupChangeDetection(data_folder)
    changeDetection = obj_dict['changeDetection']
    changeDetection.run()
