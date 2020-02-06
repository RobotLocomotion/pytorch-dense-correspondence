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
import dense_correspondence.loss_functions.utils as loss_utils
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence.training.train_heatmap import train_dense_descriptors



DATASET_NAME = "caterpillar_9"
MODEL_NAME = pdc_utils.get_current_YYYY_MM_DD_hh_mm_ss() + "_resnet50_" + "_dataset_" + DATASET_NAME


DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), "pdc")
OUTPUT_DIR = os.path.join(DATA_ROOT, "dev/experiments/heatmap/trained_models", MODEL_NAME)
print("OUTPUT_DIR", OUTPUT_DIR)

episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/logs_proto")
episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
                                                          'config/dense_correspondence/dataset/single_object/caterpillar_9_episodes.yaml'))
multi_episode_dict = SpartanEpisodeReader.load_dataset(episode_list_config,
                                                      episodes_root)

# placeholder for now
config_file = os.path.join(getDenseCorrespondenceSourceDir(),
                           'config/dense_correspondence/global/drake_sim_dynamic.yaml')
config = getDictFromYamlFilename(config_file)
config['dataset']['name'] = DATASET_NAME


train_dense_descriptors(config,
                        train_dir=OUTPUT_DIR,
                        multi_episode_dict=multi_episode_dict,
                        verbose=False)