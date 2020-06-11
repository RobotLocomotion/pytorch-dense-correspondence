import random
import numpy as np
import logging
import time

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# pdc
from dense_correspondence.correspondence_tools import correspondence_finder
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence_manipulation.utils import torch_utils
import dense_correspondence_manipulation.utils.constants as constants
from dense_correspondence.dataset.utils import make_dynamic_episode_index
from dense_correspondence.correspondence_tools.correspondence_finder import compute_correspondence_data
from dense_correspondence.dataset.image_augmentation import ImageAugmentation


class DynamicDrakeSimDataset(data.Dataset):
    """
    Should be able to consume any EpisodeContainer class

    note: (manuelli) we should rename this to CorrespondenceDataset
    """

    def __init__(self,
                 config, # dict
                 episodes, # dict, values of type EpisodeReader
                 phase="train",
                 ):


        assert phase in ["train", "valid"]

        self._config = config
        self._episodes = episodes
        self._phase = phase
        self.verbose = False
        self.debug = False



        self._data_augmentation_enabled = self.config['dataset']['data_augmentation'] and (self._phase == "train")
        self._data_augmenter = ImageAugmentation(enabled=self._data_augmentation_enabled)
        self._data_augmenter.make_default()

        self.initialize()

    def _getitem(self,
                 episode, # EpisodeReader
                 idx, # int
                 camera_name_a, # str
                 camera_name_b, # str
                 idx_a=None,
                 idx_b=None,
                 ):


        if idx_a is None:
            idx_a = idx

        if idx_b is None:
            idx_b = idx


        data_a = episode.get_image_data(camera_name_a, idx_a)
        data_b = episode.get_image_data(camera_name_b, idx_b)



        # if it failed this will be None
        c = self._config['dataset']
        correspondence_data = \
            compute_correspondence_data(data_a,
                                        data_b,
                                        N_matches=c['N_matches'],
                                        N_masked_non_matches=c['N_masked_non_matches'],
                                        N_background_non_matches=c['N_background_non_matches'],
                                        sample_matches_only_off_mask=c['sample_matches_only_off_mask'],
                                        rgb_to_tensor_transform=self.rgb_to_tensor_transform,
                                        device='CPU',
                                        verbose=self.verbose,
                                        )




        self._data_augmenter.augment_image(data_a)
        self._data_augmenter.augment_image(data_b)

        # add rgb_tensor to data_a/data_b
        data_a['rgb_tensor'] = self.rgb_to_tensor_transform(data_a['rgb'])
        data_b['rgb_tensor'] = self.rgb_to_tensor_transform(data_b['rgb'])

        # returns a dict whose values are tensors
        # if it was invalid it returns None
        return correspondence_data

    def __getitem__(self, item_idx):
        """
        For use by a torch DataLoader. Finds entry in index, calls the internal _getitem method
        :param item_idx:
        :type item_idx:
        :return:
        :rtype:
        """

        # if epoch_size != None, then we need to randomize item_idx
        if self._epoch_size is not None:
            item_idx = random.randint(0, len(self.index)-1)

        entry = self.index[item_idx]
        episode = self._episodes[entry['episode_name']]
        idx = entry['idx_a']
        data = self._getitem(episode,
                             idx,
                             entry['camera_name_a'],
                             entry['camera_name_b'],
                             idx_a=entry['idx_a'],
                             idx_b=entry['idx_b'])

        # pad it so can be used in batch with DataLoader
        N_matches = self._config['dataset']['N_matches']
        N_masked_non_matches = self._config['dataset']['N_masked_non_matches']
        N_background_non_matches = self._config['dataset']['N_background_non_matches']
        correspondence_finder.pad_correspondence_data(data,
                                                      N_matches=N_matches,
                                                      N_masked_non_matches=N_masked_non_matches,
                                                      N_background_non_matches=N_background_non_matches,
                                                      verbose=False)

        return data

    def __len__(self):
        if self._epoch_size is not None:
            return min(self._epoch_size, len(self.index))
        else:
            return len(self.index)


    def initialize(self):
        """
        Initialize
        - setup train/valid splits
        - setup rgb --> tensor transform
        :return:
        :rtype:
        """
        # setup train/valid splits
        self.set_test_train_splits()

        # rgb --> tensor transform
        self._initialize_rgb_image_to_tensor()

    def set_test_train_splits(self):
        """
        Divides episodes into test/train splits
        :return:
        :rtype:
        """
        episode_names = list(self._episodes.keys())
        episode_names.sort()  # to make sure train/test splits are deterministic

        n_train = int(len(episode_names) * self._config['train']['train_valid_ratio'])
        n_valid = len(episode_names) - n_train

        self._train_episode_names = episode_names[0:n_train]
        self._valid_episode_names = episode_names[n_train:]

        self._train_index = self.make_index(self._train_episode_names)
        self._valid_index = self.make_index(self._valid_episode_names)
        self._all_index = self._train_index + self._valid_index

        # setup the epoch size
        self._epoch_size = None
        if "epoch_size" in self._config['dataset'] and self._config['dataset']["epoch_size"][self._phase] > 0:
            print("setting fixed epoch size")
            self._epoch_size = self._config['dataset']["epoch_size"][self._phase]
        else:
            self._epoch_size = len(self.index)


    def make_index(self,
                   episode_names):
        index = []

        camera_names = None
        if 'camera_names' in self.config['dataset'] and len(self.config['dataset']['camera_names']) > 0:
            camera_names = self.config['dataset']['camera_names']


        for name in episode_names:
            episode = self._episodes[name]
            index.extend(episode.make_index(episode_name=name, camera_names=camera_names))

        return index

    @property
    def index(self):
        if self._phase == "train":
            return self._train_index
        elif self._phase == "valid":
            return self._valid_index
        elif self._phase == "all":
            return self._all_index
        else:
            raise ValueError

    @property
    def episode_names(self):
        if self._phase == "train":
            return self._train_episode_names
        elif self._phase == "valid":
            return self._valid_episode_names
        else:
            raise ValueError("unknown phase")

    @property
    def episodes(self):
        """
        Return the dict containing all the episodes
        :return:
        :rtype:
        """
        return self._episodes

    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:r
        """
        self._rgb_image_to_tensor = torch_utils.make_default_image_to_tensor_transform()

    @property
    def rgb_to_tensor_transform(self):
        return self._rgb_image_to_tensor

    @rgb_to_tensor_transform.setter
    def rgb_to_tensor_transform(self, val):
        self._rgb_image_to_tensor = val

    @property
    def config(self):
        return self._config

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


