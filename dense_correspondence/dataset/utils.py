import numpy as np
from PIL import Image



def make_dynamic_episode_index(episode_names,  # list of str
                               episode_dict,  # dict with values EpisodeReader
                               ):
    """
    List that of scene_names and indices
    :param episode_names:
    :type episode_names:
    :return:
    :rtype:
    """


    counter = 0
    index = []

    for name in episode_names:
        episode = episode_dict[name]
        camera_names = episode.camera_names
        for idx in range(len(episode)):
            for camera_name_a, camera_name_b in zip(camera_names, camera_names):
                entry = {'episode_name': name,
                         'idx': idx,
                         'camera_name_a': camera_name_a,
                         'camera_name_b': camera_name_b}

                index.append(entry)

    return index


def collate_fn_filter_none_type(data_samples):
    """
    Filters out None types from dataloader
    :param data_samples:
    :type data_samples:
    :return:
    :rtype:
    """

    raise NotImplementedError("WIP")

    # following https://github.com/pytorch/pytorch/issues/1137
    # also https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    # and https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    filtered_data_samples = []
    for sample in data_samples:
        if sample is not None:
            filtered_data_samples.append(sample)

    return default_collate(filtered_data_samples)

def camera_K_matrix_from_dict(camera_info,
                              ): # np.array 3 x 3
    K = np.array(camera_info['camera_matrix']['data'])
    K = K.reshape([3,3])
    return K

def load_rgb_image_from_file(filename):
    rgb_PIL = Image.open(filename).convert('RGB')
    rgb_numpy = np.array(rgb_PIL)
    return rgb_PIL, rgb_numpy


def load_depth_int16_from_file(filename):
    depth_PIL = Image.open(filename) # should have int16 dtype
    depth_np = np.array(depth_PIL)
    return depth_PIL, depth_np

def load_mask_image_from_file(filename):
    mask_PIL = Image.open(filename)
    mask_np = np.array(mask_PIL)
    return mask_PIL, mask_np

