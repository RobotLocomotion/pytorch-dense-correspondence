import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


# torch
import torch




# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence.correspondence_tools.correspondence_finder import reproject_pixels
from dense_correspondence.correspondence_tools import correspondence_plotter
from dense_correspondence.correspondence_tools.correspondence_finder import compute_correspondence_data, pad_correspondence_data
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename


import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader
from dense_correspondence.training.train_integral_heatmap_3d import train_dense_descriptors
from dense_correspondence_manipulation.utils import dev_utils


DATASET_NAME = "real_push_box"
DATASET_NAME = "real_right_many_shoes_se2"
DATASET_NAME = "real_right_boot"

multi_episode_dict = None
if DATASET_NAME == "real_push_box":
    multi_episode_dict = dev_utils.load_push_box_episodes()
elif DATASET_NAME == "real_right_many_shoes_se2":
    multi_episode_dict = dev_utils.load_shoe_imitation_episodes()
elif DATASET_NAME == "real_right_boot":
    multi_episode_dict = dev_utils.load_flip_right_boot_episodes()
else:
    raise ValueError("unknown dataset type")

MODEL_NAME = pdc_utils.get_current_YYYY_MM_DD_hh_mm_ss() + "_3D_loss" + "_resnet50_" + "_dataset_" + DATASET_NAME

DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), "pdc")
OUTPUT_DIR = os.path.join(DATA_ROOT, "dev/experiments/heatmap_dynamic/trained_models", MODEL_NAME)
print("OUTPUT_DIR", OUTPUT_DIR)

# placeholder for now
config = dev_utils.load_integral_heatmap_3d_config()
config['dataset']['name'] = DATASET_NAME


train_dense_descriptors(config,
                        train_dir=OUTPUT_DIR,
                        multi_episode_dict=multi_episode_dict,
                        verbose=False)