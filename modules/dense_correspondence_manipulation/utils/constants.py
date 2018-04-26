import os
import dense_correspondence_manipulation.utils.utils as utils

CHANGE_DETECTION_CONFIG_FILE = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection.yaml')

CHANGE_DETECTION_BACKGROUND_SUBTRACTION_CONFIG_FILE = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection_background_subtraction.yaml')

BACKGROUND_SCENE_DATA_FOLDER = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'data_volume', 'pdc','logs_proto', '14_background')

DEPTH_IM_SCALE = 1000.0   # This represents that depth images are saved as uint16, where the integer value
                          # is depth in millimeters.  So this scale just converts millimeters to meters.

DEPTH_IM_RESCALE = 4000.0 # Only for visualizaiton purposes

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD_DEV = [0.229, 0.224, 0.225]