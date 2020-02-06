

class EpisodeReader(object):
    """
    Defines the episode reader API. To incorporate data from a new source you must simply
    write the appropriate EpisodeReader class
    """

    def __init__(self):
        pass

    @property
    def length(self):
        raise NotImplementedError

    @property
    def camera_names(self):
        """
        return list of camera names
        :return:
        :rtype:
        """
        raise NotImplementedError

    def camera_pose(self,
                    camera_name,  # str
                    idx=None,  # not needed in this dataset
                    ):  # np.array [4,4] homogeneous transform T_world_camera
        raise NotImplementedError

    def camera_K_matrix(self,
                        camera_name,  # str
                        ):  # camera matrix shape [3,3]
        raise NotImplementedError

    def get_image(self,
                  camera_name,  # str
                  idx,  # int
                  type,  # string ["rgb", "depth", "depth_float32", "depth_int16", "mask", "label"]
                  ):
        raise NotImplementedError

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
                }

        :param camera_name:
        :type camera_name:
        :param idx:
        :type idx:
        :return:
        :rtype:
        """

        raise NotImplementedError

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