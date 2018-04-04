import os
import dense_correspondence_manipulation.utils.utils as utils

DEPTH_IM_RESCALE = 4000.0
CHANGE_DETECTION_CONFIG_FILE = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection.yaml')

CHANGE_DETECTION_BACKGROUND_SUBTRACTION_CONFIG_FILE = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1', 'change_detection_background_subtraction.yaml')


BACKGROUND_SCENE_DATA_FOLDER = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'data_volume', 'test', '14_background')

DEPTH_IM_SCALE = 1000.0 #
