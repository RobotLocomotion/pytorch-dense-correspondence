from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType

import os
import logging
import glob
import random

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

class SpartanDataset(DenseCorrespondenceDataset):

    PADDED_STRING_WIDTH = 6

    def __init__(self, debug=False, mode="train", config=None):

        if config is None:
            self.logs_root_path = self.load_from_config_yaml("relative_path_to_spartan_logs")
            train_test_config = os.path.join(os.path.dirname(__file__), "train_test_config", "0001_drill_test.yaml")
            self.set_train_test_split_from_yaml(train_test_config)
        else:
            # assume config has already been parsed
            self._config = config
            self.logs_root_path = utils.convert_to_absolute_path(self._config['logs_root_path'])
            self.set_train_test_split_from_yaml(self._config)


        self._pose_data = dict()

        
        if mode == "test":
            self.set_test_mode()

        self.init_length()
        print "Using SpartanDataset:"
        print "   - in", self.mode, "mode"
        print "   - number of scenes:", len(self.scenes)
        print "   - total images:    ", self.num_images_total

        DenseCorrespondenceDataset.__init__(self, debug=debug)

    def get_pose(self, rgb_filename):
        scene_directory = rgb_filename.split("images")[0]
        index = self.get_index(rgb_filename)
        pose_list = self.get_pose_list(scene_directory, "images.posegraph")
        pose_elasticfusion = self.get_pose_from_list(int(index), pose_list)
        pose_matrix4 = self.elasticfusion_pose_to_homogeneous_transform(pose_elasticfusion)
        return pose_matrix4

    def get_full_path_for_scene(self, scene_name):
        """
        Returns the full path to the processed logs folder
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        return os.path.join(self.logs_root_path, scene_name, 'processed')

    def load_all_pose_data(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        :return:
        :rtype:
        """

        for scene in self.scenes:
            self.get_pose_data(scene)

    def get_pose_data(self, scene_name):
        if scene_name not in self._pose_data:
            logging.info("Loading pose data for scene %s" %(scene_name) )
            pose_data_filename = os.path.join(self.get_full_path_for_scene(scene_name),
                                              'images', 'pose_data.yaml')
            self._pose_data[scene_name] = utils.getDictFromYamlFilename(pose_data_filename)

        return self._pose_data[scene_name]


    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """

        :param scene_name: str
        :param img_idx: int
        :return: 4 x 4 numpy array
        """
        idx = int(idx)
        scene_pose_data = self.get_pose_data(scene_name)
        pose_data = scene_pose_data[idx]['camera_to_world']
        return utils.homogenous_transform_from_dict(pose_data)


    def get_pose_from_list(self, index, pose_list):
        pose = pose_list[index]
        pose = [float(x) for x in pose[1:]]
        return pose

    def get_index(self, rgb_filename):
        prefix = rgb_filename.split("_rgb")[0]
        return prefix.split("images/")[1]

    def get_mask_filename(self, rgb_filename):
        images_masks_dir = os.path.join(os.path.dirname(os.path.dirname(rgb_filename)), "image_masks")
        index = self.get_index(rgb_filename)
        mask_filename = images_masks_dir+"/"+index+"_mask.png"
        return mask_filename

    def get_image_filename(self, scene_name, img_index, image_type):
        """
        Get the image filename for that scene and image index
        :param scene_name: str
        :param img_index: str or int
        :param image_type: ImageType
        :return:
        """

        # @todo(manuelli) check that scene_name actually exists
        scene_directory = self.get_full_path_for_scene(scene_name)



        if image_type == ImageType.RGB:
            images_dir = os.path.join(scene_directory, 'images')
            file_extension = "_rgb.png"
        elif image_type == ImageType.DEPTH:
            images_dir = os.path.join(scene_directory, 'rendered_images')
            file_extension = "_depth.png"
        elif image_type == ImageType.MASK:
            images_dir = os.path.join(scene_directory, 'image_masks')
            file_extension = "_mask.png"
        else:
            raise ValueError("unsupported image type")

        if isinstance(img_index, int):
            img_index = utils.getPaddedString(img_index, width=SpartanDataset.PADDED_STRING_WIDTH)

        scene_directory = self.get_full_path_for_scene(scene_name)
        if not os.path.isdir(scene_directory):
            raise ValueError("scene_name = %s doesn't exist" %(scene_name))

        return os.path.join(images_dir, img_index + file_extension)

    def get_camera_intrinsics(self, scene_name=None):
        """
        Returns the camera matrix for that scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """

        if scene_name is None:
            scene_directory = self.get_random_scene_directory()
        else:
            scene_directory = os.path.join(self.logs_root_path, scene_name)

        camera_info_file = os.path.join(scene_directory, 'processed', 'images', 'camera_info.yaml')
        return CameraIntrinsics.from_yaml_file(camera_info_file)

    def get_random_image_index(self, scene_name):
        """
        Returns a random image index from a given scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        pose_data = self.get_pose_data(scene_name)
        image_idxs = pose_data.keys() # list of integers
        random.choice(image_idxs)
        random_idx = random.choice(image_idxs)
        return random_idx

    @staticmethod
    def make_default_10_scenes_drill():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                   'dataset',
                                   'spartan_dataset_masked.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset






