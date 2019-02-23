from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType
from dynamic_spartan_dataset import DynamicSpartanDataset

import os
import numpy as np
import logging
import glob
import random
import copy

import torch

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision
from torchvision import transforms

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics


import dense_correspondence_manipulation.utils.constants as constants


utils.add_dense_correspondence_to_python_path()
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation

from dense_correspondence.dataset.scene_structure import SceneStructure



class DynamicTimeContrastDataset(DynamicSpartanDataset):

    def __init__(self, debug=False, mode="train", config=None, config_expanded=None):
        """
        See DynamicSpartanDataset for documentation
        """
        DynamicSpartanDataset.__init__(self, debug=debug, mode=mode, config=config, config_expanded=config_expanded)
        # HACK SINCE THIS DOESN'T MATTER
        self.num_images_total = 1000 

    def __len__(self):
        return self.num_images_total

    def __getitem__(self, index):
        """
        This is different for time contrast than for dense correspondence
        
        First version will simply return three images:
        - anchor
        - positive
        - negative

        To do that we simply need,
        - choose random scene
        - choose random index
        - return two images from that index for anchor and positive
        - randomly select a different index for the negative
        - return an image from that index for negative
        """
        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)
        idx_anchor = self.get_random_image_index(scene_name)
        camera_num_anchor, camera_num_positive = self.get_random_camera_nums_for_image_index(idx_anchor)

        rgb_anchor = self.get_rgb_image(self.get_image_filename(scene_name, camera_num_anchor, idx_anchor, ImageType.RGB))
        rgb_positive = self.get_rgb_image(self.get_image_filename(scene_name, camera_num_positive, idx_anchor, ImageType.RGB))

        idx_negative = self.get_different_enough_index(scene_name, idx_anchor)
        rand_camera_num = random.choice([0,1])
        rgb_negative = self.get_rgb_image(self.get_image_filename(scene_name, rand_camera_num, idx_negative, ImageType.RGB))

        rgb_list_torch = [self.rgb_image_to_tensor(x) for x in [rgb_anchor, rgb_positive, rgb_negative]]

        metadata = dict()
        metadata["object_id"] = object_id
        metadata["object_id_int"] = sorted(self._single_object_scene_dict.keys()).index(object_id)
        metadata["scene_name"] = scene_name
        metadata["idx_anchor"] = idx_anchor
        metadata["idx_negative"] = idx_negative

        if self.debug:
            self.debug_show_images(rgb_list_torch)
        return metadata, rgb_list_torch

    def debug_show_images(self, rgb_list_torch):
        import matplotlib.pyplot as plt
        for rgb_torch in rgb_list_torch:
            rgb_numpy = rgb_torch.permute(1,2,0).numpy()
            plt.imshow(rgb_numpy)
            plt.show()

    def get_different_enough_index(self, scene_name, index):
        scene_directory = self.get_full_path_for_scene(scene_name)
        state_info_filename = os.path.join(scene_directory, "states.yaml")
        state_info_dict = utils.getDictFromYamlFilename(state_info_filename)
        length_before = len(state_info_dict)
        image_idxs = state_info_dict.keys() # list of integers
        time_margin = 15
        negative_time = max(0,                 index-time_margin)
        positive_time = min(len(image_idxs)-1, index+time_margin)
        del image_idxs[negative_time:positive_time+1]
        assert length_before == len(state_info_dict)
        return random.choice(image_idxs)


