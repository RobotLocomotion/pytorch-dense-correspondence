import abc

class EpisodeReader(abc.ABC):
    """
    Defines the episode reader API. To incorporate data from a new source you must simply
    write the appropriate EpisodeReader class
    """

    def __init__(self):
        pass

    @property
    def length(self):
        """
        Note: I don't think we actually need this method, make_index sort of plays this role
        :return:
        :rtype:
        """
        raise NotImplementedError

    @property
    def camera_names(self):
        """
        return list of camera names
        :return:
        :rtype:
        """
        raise NotImplementedError

    @property
    def episode_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def camera_pose(self,
                    camera_name,  # str
                    idx,  # not needed in this dataset
                    ):  # np.array [4,4] homogeneous transform T_world_camera
        return
        # raise NotImplementedError

    @abc.abstractmethod
    def camera_K_matrix(self,
                        camera_name,  # str
                        ):  # camera matrix shape [3,3]
        return
        # raise NotImplementedError

    @abc.abstractmethod
    def get_image(self,
                  camera_name,  # str
                  idx,  # int
                  type,  # string ["rgb", "depth", "depth_float32", "depth_int16", "mask", "label"]
                  ):
        """
        For a depth_int16 or depth_float32 a value of 0 indicates a pixel where no valid
        depth value exists. It should not be used in any computations.
        """
        return

    @abc.abstractmethod
    def get_image_data(self,
                       camera_name,
                       idx):
        """
        Optional returns a dict with image data aggregated
        Some fields may be ommitted (e.g. 'label')

        return {'rgb': rgb,
                'label': label,
                'mask': mask,
                'depth_float32': depth_float32, # meters
                'depth_int16': depth_int16, # millimeters
                'K': K,
                'T_world_camera': T_W_C,
                'episode_name': episode_name,
                'camera_name': camera_name,
                'idx': idx
                }

        :param camera_name:
        :type camera_name:
        :param idx:
        :type idx:
        :return:
        :rtype:
        """

        return
        # raise NotImplementedError

    def make_index(self,
                   episode_name=None,
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
        raise NotImplementedError

    def make_single_image_index(self,
                                episode_name=None,  # name to give this episode
                                camera_names=None,  # (optional) list[str]
                                ):
        """
        Makes index to iterate through all the images.
        A single entry in the list is a dict of the form

        entry = {'episode_name': episode_name,
                 'idx': idx,
                 'camera_name': camera_name,
                 }
        :param episode_name:
        :type episode_name:
        :return:
        :rtype:
        """
        raise NotImplementedError

    def get_image_dimensions(self,
                             camera_name, # str: not needed
                             ): # (int, int): Width and Height
        """
        Get width and heigh of image corresponding to this camera
        :param camera_name:
        :type camera_name:
        :return:
        :rtype:
        """

        raise NotImplementedError

    def get_image_key(self,
                      camera_name,
                      idx,
                      type):
        key = "/".join([camera_name, str(idx), type])
        return key

    def get_image_key_tree(self,
                           camera_name,
                           idx):
        """
        Used when saving descriptor images to disk
        :param camera_name:
        :type camera_name:
        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        return [camera_name, str(idx)]


    def get_image_data_specified_in_config(self,
                                           camera_name,
                                           idx,
                                           image_config,  # dict
                                           ):
        """
        Get image data specified in a config
        :param camera_name:
        :type camera_name:
        :param idx:
        :type idx:
        :param image_config:
        :type image_config:
        :return:
        :rtype:
        """

        return_data = dict()

        image_types = ['rgb', 'mask', 'depth_int16', 'descriptor',
                       'descriptor_keypoints']

        for image_type in image_types:
            flag = False # whether to grab this image type

            if image_type in image_config:
                if image_config[image_type]:
                    flag = True

            if flag:
                return_data[image_type] = self.get_image(camera_name, idx, image_type)

        T_W_C = self.camera_pose(camera_name, idx)
        K = self.camera_K_matrix(camera_name)

        return_data['T_W_C'] = T_W_C
        return_data['camera_name'] = camera_name
        return_data['idx'] = idx
        return_data['K'] = K

        return return_data