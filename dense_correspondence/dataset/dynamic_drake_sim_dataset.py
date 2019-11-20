import random
import numpy as np
import logging


import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# pdc
from dense_correspondence.correspondence_tools import correspondence_finder
from dense_correspondence_manipulation.utils import utils as pdc_utils
import dense_correspondence_manipulation.utils.constants as constants


class DynamicDrakeSimDataset(data.Dataset):
    """
    Should be able to consume any EpisodeContainer class
    """

    def __init__(self,
                 config, # dict
                 episodes, # dict, values of type EpisodeReader
                 phase="train"):

        self._config = config
        self._episodes = episodes
        self._phase = phase
        self._initialize_rgb_image_to_tensor()

    def _getitem(self, idx):
        # local version

        # returns a dict whose values are tensors
        return {"rgb_a": image_a_rgb,
                "rgb_b": image_b_rgb,
                "matches_a": matches_a,
                "matches_b": matches_b,
                "masked_non_matches_a": masked_non_matches_a,
                "masked_non_matches_b": masked_non_matches_b,
                "metadata": metadata,
                }

    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_MEAN

        # return self.config["image_normalization"]["mean"]

        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_STD_DEV

        # return self.config["image_normalization"]["std_dev"]

        return constants.DEFAULT_IMAGE_STD_DEV

    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image or numpy.ndarray to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """

        return self._rgb_image_to_tensor(img)

    def compute_correspondences(self,
                                data_a, # dict
                                data_b, # dict
                                sample_matches_only_off_mask=True):

        # return data
        return_data = dict()

        image_width = data_a['rgb'].shape[1]
        image_height = data_a['rgb'].shape[0]

        img_size = np.size(data_a['rgb'])
        min_mask_size = 0.01*img_size


        # skip if not enough pixels in mask
        if (np.sum(data_a['mask']) < min_mask_size) or (np.sum(data_b['mask']) < min_mask_size):
            logging.info("not enough pixels in mask, skipping")
            return False, return_data

        # set the mask for correspondences
        if sample_matches_only_off_mask:
            correspondence_mask = np.asarray(data_a['mask'])
        else:
            correspondence_mask = None

        uv_a, uv_b, uv_a_not_detected= \
            correspondence_finder.batch_find_pixel_correspondences(img_a_depth=data_a['depth_16U'],
                                                                    img_a_pose=data_a['T_world_camera'],
                                                                    img_b_depth=data_b['depth_16U'],
                                                                    img_b_pose=data_b['T_world_camera'],
                                                                   img_a_mask=correspondence_mask,
                                                                    K_a=data_a['K'],
                                                                    K_b=data_b['K'],
                                                                   matching_type="with_detections", # not sure what this does
                                                                    )


        # perform photometric check
        matches_a = pdc_utils.flatten_uv_tensor(uv_a, image_width)
        matches_b = pdc_utils.flatten_uv_tensor(uv_b, image_width)

        # need to be [D,H,W] torch.FloatTensors that have already
        # been normalized
        rgb_tensor_a = self.rgb_image_to_tensor(data_a['rgb'])
        rgb_tensor_b = self.rgb_image_to_tensor(data_b['rgb'])

        matches_a, matches_b = correspondence_finder.photometric_check(rgb_tensor_a, rgb_tensor_b, matches_a, matches_b)


        # compute non-correspondences
        non_matches_a = None
        non_matches_b = None
        masked_non_matches_a = None
        masked_non_matches_b = None


        # data augmentation should happen elsewhere
        metadata = dict()
        return_data = {'data_a': data_a,
                       'data_b': data_b,
                       'matches_a': matches_a,
                       'matches_b': matches_b,
                       'non_matches_a': non_matches_a,
                       'non_matches_b': non_matches_b,
                       'masked_non_matches_a': masked_non_matches_a,
                       'masked_non_matches_b': masked_non_matches_b,
                       'metadata': metadata}

        return True, return_data








