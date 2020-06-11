import os
import copy
import random

from dense_correspondence.dataset.episode_reader import EpisodeReader
import dense_correspondence.dataset.utils as dataset_utils
from dense_correspondence_manipulation.utils.utils import getDictFromYamlFilename
import dense_correspondence_manipulation.utils.utils as utils


class SpartanEpisodeReader(EpisodeReader):
    DEFAULT_CAMERA_NAME = "camera_0"
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
        self._camera_name = SpartanEpisodeReader.DEFAULT_CAMERA_NAME
        self._metadata = metadata

        # load camera_info
        self.camera_matrix_dict = dict()
        self.camera_matrix_dict[self._camera_name] = self.camera_K_matrix()

        # load pose data
        self._pose_data = getDictFromYamlFilename(self.pose_data_file)
        self._indices = list(self._pose_data.keys())
        self._indices.sort()

    @property
    def config(self):
        return self._config

    @property
    def camera_info_file(self):
        return os.path.join(self._root_dir, 'images', 'camera_info.yaml')

    @property
    def pose_data_file(self):
        return os.path.join(self._root_dir, 'images', 'pose_data.yaml')

    @property
    def camera_names(self):
        return [self._camera_name]

    @property
    def episode_name(self):
        return self._name

    def camera_K_matrix(self,
                        camera_name=None,
                        ):

        camera_info = getDictFromYamlFilename(self.camera_info_file)
        K = dataset_utils.camera_K_matrix_from_dict(camera_info)
        return K

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

    def get_image_filename(self, img_index, image_type):
        """
        Get the image filename for that scene and image index
        :param scene_name: str
        :param img_index: str or int
        :param image_type: ImageType
        :return:
        """

        if image_type == "rgb":
            images_dir = os.path.join(self._root_dir, 'images')
            file_extension = "_rgb.png"
        elif image_type == "depth_int16":
            images_dir = os.path.join(self._root_dir, 'rendered_images')
            file_extension = "_depth.png"
        elif image_type == "raw_depth_int16":
            images_dir = os.path.join(self._root_dir, 'images')
            file_extension = "_depth.png"
        elif image_type == "mask":
            images_dir = os.path.join(self._root_dir, 'image_masks')
            file_extension = "_mask.png"
        else:
            raise ValueError("unsupported image type")

        if isinstance(img_index, int):
            img_index = utils.getPaddedString(img_index, width=SpartanEpisodeReader.PADDED_STRING_WIDTH)

        return os.path.join(images_dir, img_index + file_extension)

    def get_rgb_image(self,
                      camera_name,
                      idx):  # np.array dtype uint8?

        filename = self.get_image_filename(idx, "rgb")
        _, rgb_np = dataset_utils.load_rgb_image_from_file(filename)
        return rgb_np

    def get_depth_image_int16(self,
                              camera_name,
                              idx):  # np.array, dtype = ?

        filename = self.get_image_filename(idx, "depth_int16")
        _, depth_int16 = dataset_utils.load_depth_int16_from_file(filename)
        return depth_int16

    def get_depth_image_float32(self,
                                camera_name,
                                idx):  # np.array, dtype=?
        raise NotImplementedError

    def get_raw_depth_image_int16(self,
                                  camera_name,
                                  idx):
        filename = self.get_image_filename(idx, "raw_depth_int16")
        _, depth_int16 = dataset_utils.load_depth_int16_from_file(filename)
        return depth_int16


    def get_mask_image(self,
                       camera_name,
                       idx):  # np.array, dtype=?
        filename = self.get_image_filename(idx, "mask")
        _, mask = dataset_utils.load_mask_image_from_file(filename)
        return mask


    def camera_pose(self,
                    camera_name,
                    idx):

        pose_dict = self._pose_data[idx]['camera_to_world']
        T_W_C = utils.homogenous_transform_from_dict(pose_dict)
        return T_W_C

    def get_image_data(self,
                       camera_name,
                       idx):

        rgb = self.get_rgb_image(camera_name, idx)
        depth_int16 = self.get_depth_image_int16(camera_name, idx)
        mask = self.get_mask_image(camera_name, idx)
        T_W_C = self.camera_pose(camera_name, idx)
        K = self.camera_matrix_dict[camera_name]

        return {'rgb': rgb,
                'depth_int16': depth_int16,
                'mask': mask,
                'T_world_camera': T_W_C,
                'K': K,
                'camera_name': camera_name,
                'idx': idx,
                'episode_name': self.episode_name,
                }

    def get_image_dimensions(self,
                             camera_name,  # str: not needed
                             ):

        camera_info = getDictFromYamlFilename(self.camera_info_file)
        W = camera_info["image_width"]
        H = camera_info["image_height"]

        return W, H

    def make_index(self,
                   episode_name=None,
                   camera_names=None, #list[str]
                   ):
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


        index = []

        if camera_names is None:
            camera_names = [self.camera_names]
        else:
            camera_names = list(set(camera_names) & set(self.camera_names))

        if len(camera_names) == 0:
            return []

        camera_name = camera_names[0]

        for idx_a in self.indices:
            for idx_b in self.indices:
                data = {'episode_name': episode_name,
                        'camera_name_a': camera_name,
                        'camera_name_b': camera_name,
                        'idx_a': idx_a,
                        'idx_b': idx_b}
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
            episode = SpartanEpisodeReader(config,
                                           episode_processed_dir,
                                           name=episode_name, )

            multi_episode_dict[episode_name] = episode


        return multi_episode_dict
