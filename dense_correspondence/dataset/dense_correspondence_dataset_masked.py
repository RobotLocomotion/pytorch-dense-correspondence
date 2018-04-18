import torch
import torch.utils.data as data

import os
import math
import yaml
import logging
import numpy as np
import random
import glob
from PIL import Image

import sys
sys.path.insert(0, '../../pytorch-segmentation-detection/vision/')
from torchvision import transforms
sys.path.append('../../pytorch-segmentation-detection/')
from pytorch_segmentation_detection.transforms import ComposeJoint
sys.path.append('../correspondence_tools/')
import correspondence_finder
import correspondence_augmentation
import dense_correspondence_manipulation.utils.utils as utils

# This implements a subclass for a data.Dataset class in PyTorch
# to load in data for dense descriptor training
#
# in particular note:
# __len__     is overloaded
# __getitem__ is overloaded
#
# For more info see:
# http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset

class ImageType:
    RGB = 0
    DEPTH = 1
    MASK = 2

class DenseCorrespondenceDataset(data.Dataset):

    def __init__(self, debug=False):
        
        self.debug = debug
        self.mode = "train"
        self.both_to_tensor = ComposeJoint(
            [
                [transforms.ToTensor(), transforms.ToTensor()]
            ])

        # Otherwise, all of these parameters should be set in
        # set_parameters_from_training_config()
        if self.debug:
            self.num_matching_attempts = 20
            self.num_non_matches_per_match = 1
      
    def __len__(self):
        return self.num_images_total
    
    def __getitem__(self, index):
        """
        The method through which the dataset is accessed for training.

        The index param is not currently used, and instead each dataset[i] is the result of
        a random sampling over:
        - random scene
        - random rgbd frame from that scene
        - random rgbd frame (different enough pose) from that scene
        - various randomization in the match generation and non-match generation procedure

        returns a large amount of variables, separated by commas.

        0th return arg: the type of data sampled (this can be used as a flag for different loss functions)
        0th rtype: string

        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width, 3)

        3rd, 4th return args: matches_a, matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)

        5th, 6th return args: non_matches_a, non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        Return values 3,4,5,6 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        """

        # pick a scene
        scene_name = self.get_random_scene_name()

        # image a
        image_a_idx = self.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)

        # image b
        image_b_idx = self.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)

        if image_b_idx is None:
            logging.info("no frame with sufficiently different pose found, returning")
            # TODO: return something cleaner than no-data
            return self.return_empty_data(image_a_rgb, image_b_rgb)

        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose, 
                                                                           image_b_depth_numpy, image_b_pose, 
                                                                           num_attempts=self.num_matching_attempts, img_a_mask=np.asarray(image_a_mask))

        if uv_a is None:
            logging.info("no matches found, returning")
            return self.return_empty_data(image_a_rgb, image_b_rgb)

        if self.debug:
            # downsample so can plot
            num_matches_to_plot = 10
            indexes_to_keep = (torch.rand(num_matches_to_plot)*len(uv_a[0])).floor().type(torch.LongTensor)
            uv_a = (torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep))
            uv_b = (torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep))

        # data augmentation
        if not self.debug:
            [image_a_rgb], uv_a                 = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb], uv_a)
            [image_b_rgb, image_b_mask], uv_b   = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_mask], uv_b)
        else: # also mutate depth just for plotting
            [image_a_rgb, image_a_depth], uv_a               = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_depth], uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_depth, image_b_mask], uv_b)
            image_a_depth_numpy = np.asarray(image_a_depth)
            image_b_depth_numpy = np.asarray(image_b_depth)

        # find non_correspondences

        if index%2:
            logging.debug("masking non-matches")
            image_b_mask = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        else:
            logging.debug("not masking non-matches")
            image_b_mask = None
            
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b, num_non_matches_per_match=self.num_non_matches_per_match, img_b_mask=image_b_mask)

        if self.debug:
            # only want to bring in plotting code if in debug mode
            import correspondence_plotter

            # Just show all images 
            uv_a_long = (torch.t(uv_a[0].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1,1), 
                     torch.t(uv_a[1].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1,1))
            uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )
            
            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy, uv_a, uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy,
                                                  uv_a_long, uv_b_non_matches_long,
                                                  use_previous_plot=(fig,axes),
                                                  circ_color='r')

        image_a_rgb, image_b_rgb = self.both_to_tensor([image_a_rgb, image_b_rgb])

        uv_a_long = (torch.t(uv_a[0].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1,1), 
                     torch.t(uv_a[1].repeat(self.num_non_matches_per_match, 1)).contiguous().view(-1,1))
        uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )

        # flatten correspondences and non_correspondences
        matches_a = uv_a[1].long()*640+uv_a[0].long()
        matches_b = uv_b[1].long()*640+uv_b[0].long()
        non_matches_a = uv_a_long[1].long()*640+uv_a_long[0].long()
        non_matches_a = non_matches_a.squeeze(1)
        non_matches_b = uv_b_non_matches_long[1].long()*640+uv_b_non_matches_long[0].long()
        non_matches_b = non_matches_b.squeeze(1)

        return "matches", image_a_rgb, image_b_rgb, matches_a, matches_b, non_matches_a, non_matches_b

    def return_empty_data(self, image_a_rgb, image_b_rgb):
        None, image_a_rgb, image_b_rgb, torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()

    def get_rgbd_mask_pose(self, scene_name, img_idx):
        """
        Returns rgb image, depth image, mask and pose.
        :param scene_name:
        :type scene_name: str
        :param img_idx:
        :type img_idx: int
        :return: rgb, depth, mask, pose
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        rgb_file = self.get_image_filename(scene_name, img_idx, ImageType.RGB)
        rgb = self.get_rgb_image(rgb_file)

        depth_file = self.get_image_filename(scene_name, img_idx, ImageType.DEPTH)
        depth = self.get_depth_image(depth_file)

        mask_file = self.get_image_filename(scene_name, img_idx, ImageType.MASK)
        mask = self.get_mask_image(mask_file)

        pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)

        return rgb, depth, mask, pose

    def get_img_idx_with_different_pose(self, scene_name, pose_a, threshold=0.2, num_attempts=10):
        """
        Try to get an image with a different pose to the one passed in. If one can't be found
        then return None
        :param scene_name:
        :type scene_name:
        :param pose_a:
        :type pose_a:
        :param threshold:
        :type threshold:
        :param num_attempts:
        :type num_attempts:
        :return: an index with a different-enough pose
        :rtype: int or None
        """

        counter = 0
        while counter < num_attempts:
            img_idx = self.get_random_image_index(scene_name)
            pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)

            diff = utils.compute_distance_between_poses(pose_a, pose)
            if diff > threshold:
                return img_idx
            counter += 1

        return None


    @staticmethod
    def load_rgb_image(rgb_filename):
        """
        Returns PIL.Image.Image
        :param rgb_filename:
        :type rgb_filename:
        :return:
        :rtype: PIL.Image.Image
        """
        return Image.open(rgb_filename).convert('RGB')

    @staticmethod
    def load_mask_image(mask_filename):
        """
        Loads the mask image, returns a PIL.Image.Image
        :param mask_filename:
        :type mask_filename:
        :return:
        :rtype: PIL.Image.Image
        """
        return Image.open(mask_filename)

    def get_rgb_image(self, rgb_filename):
        """
        :param depth_filename: string of full path to depth image
        :return: PIL.Image.Image, in particular an 'RGB' PIL image
        """
        return Image.open(rgb_filename).convert('RGB')

    def get_rgb_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns an rgb image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.RGB)
        return self.get_rgb_image(img_filename)

    def get_depth_image(self, depth_filename):
        """
        :param depth_filename: string of full path to depth image
        :return: PIL.Image.Image
        """
        return Image.open(depth_filename)

    def get_depth_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns a depth image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.DEPTH)
        return self.get_depth_image(img_filename)

    def get_mask_image(self, mask_filename):
        """
        :param mask_filename: string of full path to mask image
        :return: PIL.Image.Image
        """
        return Image.open(mask_filename)

    def get_mask_image_from_scene_name_and_idx(self, scene_name, img_idx):
        """
        Returns a depth image given a scene_name and image index
        :param scene_name:
        :param img_idx: str or int
        :return: PIL.Image.Image
        """
        img_filename = self.get_image_filename(scene_name, img_idx, ImageType.MASK)
        return self.get_mask_image(img_filename)

    def get_image_filename(self, scene_name, img_index, image_type):
        raise NotImplementedError("Implement in superclass")

    def load_all_pose_data(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        :return:
        :rtype:
        """
        raise NotImplementedError("subclass must implement this method")

    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """

        :param scene_name: str
        :param img_idx: int
        :return: 4 x 4 numpy array
        """
        raise NotImplementedError("subclass must implement this method")

    # this function cowbody copied from:
    # https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    def quaternion_matrix(self, quaternion):
        _EPS = np.finfo(float).eps * 4.0
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [                0.0,                 0.0,                 0.0, 1.0]])

    def elasticfusion_pose_to_homogeneous_transform(self, lf_pose):
        homogeneous_transform = self.quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0,3] = lf_pose[0]
        homogeneous_transform[1,3] = lf_pose[1]
        homogeneous_transform[2,3] = lf_pose[2]
        return homogeneous_transform

    def get_pose_list(self, scene_directory, pose_list_filename):
        posegraph_filename = os.path.join(scene_directory, pose_list_filename)
        with open(posegraph_filename) as f:
            content = f.readlines()
        pose_list = [x.strip().split() for x in content]
        return pose_list

    def get_full_path_for_scene(self, scene_name):
        raise NotImplementedError("subclass must implement this method")

    def get_random_scene_name(self):
        """
        Returns a random scene_name
        The result will depend on whether we are in test or train mode
        :return:
        :rtype:
        """
        return random.choice(self.scenes)

    def get_random_image_index(self, scene_name):
        """
        Returns a random image index from a given scene
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        raise NotImplementedError("subclass must implement this method")

    def get_random_scene_directory(self):
        scene_name = self.get_random_scene_name()
        # can later add biases for scenes, for example based on # images?
        scene_directory = self.get_full_path_for_scene(scene_name)
        return scene_directory

    def init_length(self):
        self.num_images_total = 0
        for scene_name in self.scenes:
            scene_directory = self.get_full_path_for_scene(scene_name)
            rgb_images_regex = os.path.join(scene_directory, "images/*_rgb.png")
            all_rgb_images_in_scene = glob.glob(rgb_images_regex)
            num_images_this_scene = len(all_rgb_images_in_scene)
            self.num_images_total += num_images_this_scene

    def load_from_config_yaml(self, key):

        this_file_path = os.path.dirname(__file__)
        yaml_path = os.path.join(this_file_path, "config.yaml")

        with open(yaml_path, 'r') as stream:
            try:
                config_dict = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        import getpass
        username = getpass.getuser()

        relative_path = config_dict[username][key]
        full_path = os.path.join(os.environ['HOME'], relative_path)
        return full_path

    def use_all_available_scenes(self):
        self.scenes = [os.path.basename(x) for x in glob.glob(self.logs_root_path+"*")]

    def set_train_test_split_from_yaml(self, yaml_config_file_full_path):
        """
        Sets self.train and self.test attributes from config file
        :param yaml_config_file_full_path:
        :return:
        """
        if isinstance(yaml_config_file_full_path, str):
            with open(yaml_config_file_full_path, 'r') as stream:
                try:
                    config_dict = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            config_dict = yaml_config_file_full_path

        self.train = config_dict["train"]
        self.test  = config_dict["test"]
        self.set_train_mode()

    def set_parameters_from_training_config(self, training_config):
        """
        Some parameters that are really associated only with training, for example
        those associated with random sampling during the training process,
        should be passed in from a training.yaml config file.

        :param training_config: a dict() holding params
        """
        self.num_matching_attempts     = training_config['training']['num_matching_attempts']
        self.num_non_matches_per_match = training_config['training']['num_non_matches_per_match']

    def set_train_mode(self):
        self.scenes = self.train
        self.mode = "train"

    def set_test_mode(self):
        self.scenes = self.test
        self.mode = "test"

    @property
    def test_scene_directories(self):
        """
        Get the list of testing scene directories
        :return: list of strings
        """
        return self.test

    @property
    def train_scene_directories(self):
        """
        Get the list of training scene directories
        :return: list of strings
        """
        return self.train
    """
    Debug
    """
    def debug_show_data(self, image_a_rgb, image_a_depth, image_a_pose,
                              image_b_rgb, image_b_depth, image_b_pose):
        plt.imshow(image_a_rgb)
        plt.show()
        plt.imshow(image_a_depth)
        plt.show()
        print "image_a_pose", image_a_pose
        plt.imshow(image_b_rgb)
        plt.show()
        plt.imshow(image_b_depth)
        plt.show()
        print "image_b_pose", image_b_pose