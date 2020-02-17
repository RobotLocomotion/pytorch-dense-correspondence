import os
import copy
import random
import numpy as np
import itertools

from dense_correspondence.dataset.episode_reader import EpisodeReader
import dense_correspondence.dataset.utils as dataset_utils
from dense_correspondence_manipulation.utils.utils import getDictFromYamlFilename
import dense_correspondence_manipulation.utils.utils as utils


class DynamicSpartanEpisodeReader(EpisodeReader):
    PADDED_STRING_WIDTH = 6

    """
    Dataset for static scene with a single camera
    """

    def __init__(self,
                 config,
                 root_dir,  # str: fullpath to 'processed' dir
                 name="",  # str
                 metadata=None,  # dict: optional metadata, e.g. object type etc.
                 ):
        EpisodeReader.__init__(self)
        self._config = config
        self._root_dir = root_dir
        self._name = name


        self._metadata = metadata

        # load camera_info
        d = utils.load_json(self.data_file)
        # convert keys to int
        self._episode_data = {int(k): v for k,v in d.items()}
        self._indices = list(self._episode_data.keys())
        self._indices.sort()
        self._initialize_camera_data()

    @property
    def config(self):
        return self._config

    @property
    def length(self):
        return len(self._indices)

    @property
    def episode_name(self):
        return self._name

    def images_dir(self, camera_name):
        """
        Directory where images are stored for this camera
        Typically of the form
        root_dir/
            images_camera_0

        :param camera_name:
        :type camera_name:
        :return:
        :rtype:
        """
        return os.path.dirname(self.get_image_filename(camera_name, 0, "rgb"))

    def camera_info_file(self, camera_name):
        """
        The camera_info.yaml file corresponding to this camera
        :param camera_name:
        :type camera_name:
        :return:
        :rtype:
        """
        return os.path.join(self.images_dir(camera_name), 'camera_info.yaml')

    def camera_K_matrix(self,
                        camera_name=None,
                        ):

        camera_info = getDictFromYamlFilename(self.camera_info_file(camera_name))
        K = dataset_utils.camera_K_matrix_from_dict(camera_info)
        return K

    @property
    def data_file(self):
        return os.path.join(self._root_dir, 'states.json')

    @property
    def camera_names(self):
        data = self._episode_data[0]
        camera_names = list(data['observations']['images'].keys())
        return camera_names

    def _initialize_camera_data(self):
        """
        Populates the self.camera_info field with the K matrices
        for each camera
        :return:
        :rtype:
        """

        self.camera_info = dict()
        for camera_name in self.camera_names:
            K = self.camera_K_matrix(camera_name)
            camera_info = getDictFromYamlFilename(self.camera_info_file(camera_name))
            T_W_C = utils.homogenous_transform_from_dict(camera_info['extrinsics'])

            d = {'K': K,
                 'T_world_camera': T_W_C,
                 'images_dir': self.images_dir(camera_name),
                 'camera_info': camera_info}

            self.camera_info[camera_name] = d


    @property
    def indices(self):
        return copy.copy(self._indices)

    def get_image(self,
                  camera_name,  # str
                  idx,  # int
                  type,  # string ["rgb", "depth", "depth_float32", "depth_int16", "mask", "label"]
                  ):

        if type == "rgb":
            return self.get_rgb_image(camera_name,
                                      idx)
        elif type == "depth_int16":
            return self.get_depth_image_int16(camera_name, idx)
        elif type == "mask":
            return self.get_mask_image(camera_name, idx)
        raise NotImplementedError

    def get_image_filename(self, camera_name, img_index, image_type):
        """
        Get the image filename for that scene and image index
        :param scene_name: str
        :param img_index: str or int
        :param image_type: ImageType
        :return:
        """
        data = self._episode_data[img_index]['observations']['images'][camera_name]

        filename = None
        if image_type == "rgb":
            filename = os.path.join(self._root_dir, data['rgb_image_filename'])
        elif image_type == "depth_int16":
            filename = os.path.join(self._root_dir, data['depth_image_filename'])
        elif image_type == "mask":
            images_dir = self.camera_info[camera_name]['images_dir']
            pad_string = utils.getPaddedString(img_index, width=DynamicSpartanEpisodeReader.PADDED_STRING_WIDTH)
            mask_filename = pad_string + "_mask.png"
            filename = os.path.join(images_dir, 'image_masks', mask_filename)
        else:
            raise ValueError("unsupported image type")


        return filename

    def get_rgb_image(self,
                      camera_name,
                      idx):  # np.array dtype uint8?

        filename = self.get_image_filename(camera_name, idx, "rgb")
        _, rgb_np = dataset_utils.load_rgb_image_from_file(filename)
        return rgb_np

    def get_depth_image_int16(self,
                              camera_name,
                              idx):  # np.array, dtype = ?

        filename = self.get_image_filename(camera_name, idx, "depth_int16")
        _, depth_int16 = dataset_utils.load_depth_int16_from_file(filename)
        return depth_int16

    def get_depth_image_float32(self,
                                camera_name,
                                idx):  # np.array, dtype=?
        raise NotImplementedError

    def get_mask_image(self,
                       camera_name,
                       idx):  # np.array, dtype=?
        filename = self.get_image_filename(camera_name, idx, "mask")
        _, mask = dataset_utils.load_mask_image_from_file(filename)
        return mask

    def get_camera_pose(self,
                        camera_name,
                        idx=None, # optional for this class
                        ):

        return np.copy(self.camera_info[camera_name]['T_world_camera'])

    def get_image_data(self,
                       camera_name,
                       idx):

        rgb = self.get_rgb_image(camera_name, idx)
        depth_int16 = self.get_depth_image_int16(camera_name, idx)
        mask = self.get_mask_image(camera_name, idx)
        T_W_C = self.get_camera_pose(camera_name, idx)
        K = self.camera_info[camera_name]['K']

        return {'rgb': rgb,
                'depth_int16': depth_int16,
                'mask': mask,
                'T_world_camera': T_W_C,
                'K': K,
                'camera_name': camera_name,
                'idx': idx,
                'episode_name': self.episode_name}

    def get_image_dimensions(self,
                             camera_name,  # str: not needed
                             ):

        camera_info = self.camera_info[camera_name]['camera_info']
        W = camera_info["image_width"]
        H = camera_info["image_height"]

        return W, H

    def make_index(self,
                   episode_name=None):
        """
        Makes the index for training, will be a list of dicts.
        A single entry is of the form. Wh

        entry = {'episode_name': episode_name,
                 'camera_name_a': camera_name_a,
                 'idx_a': idx_a,
                 'camera_name_b': camera_name_b,
                 'idx_b': idx_b
                 }

        :return: list[dict]
        :rtype:
        """
        if episode_name is None:
            episode_name = self._name

        keys = list(self._episode_data.keys())

        index = []
        camera_names = self.camera_names

        for key in keys:
            for camera_name_a, camera_name_b in itertools.product(camera_names, camera_names):
                # don't include ones that are the same camera (at least for now . . . )
                if camera_name_a == camera_name_b:
                    continue

                data = {'episode_name': episode_name,
                        'camera_name_a': camera_name_a,
                        'camera_name_b': camera_name_b,
                        'idx_a': key,
                        'idx_b': key}
                index.append(data)

        # randomly shuffle this index
        random.shuffle(index)
        return index

    @staticmethod
    def load_dataset(config,  # dict, e.g. caterpillar_9_epsiodes.yaml
                     episodes_root,  # str: root of where all the logs are stored
                     ):

        multi_episode_dict = dict()
        for episode_name in config['episodes']:
            episode_processed_dir = os.path.join(episodes_root, episode_name, "processed")
            episode = DynamicSpartanEpisodeReader(config,
                                           episode_processed_dir,
                                           name=episode_name, )

            multi_episode_dict[episode_name] = episode


        return multi_episode_dict
