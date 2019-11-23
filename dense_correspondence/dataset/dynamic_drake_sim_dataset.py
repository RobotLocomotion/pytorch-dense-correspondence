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

    @property
    def rgb_to_tensor_transform(self):
        return self._rgb_image_to_tensor

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


