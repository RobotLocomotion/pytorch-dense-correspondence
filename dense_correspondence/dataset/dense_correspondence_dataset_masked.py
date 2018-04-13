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

# This implements an abstract Dataset class in PyTorch
# to load in LabelFusion data (labelfusion.csail.mit.edu)
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

        self.tensor_transform = ComposeJoint(
                [
                    [transforms.ToTensor(), None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(x).long()) ]
                ])
      
    def __len__(self):
        return self.num_images_total
    
    def __getitem__(self, index):
        dtype_long = torch.LongTensor

        # pick a scene
        scene_name = self.get_random_scene_name()

        # image a
        image_a_idx = self.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)

        # image b
        image_b_idx = self.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)

        if image_b_idx is None:
            logging.info("no frame with sufficiently different pose found, returning")
            print "no frame with sufficiently different pose found, returning"
            return "matches", image_a_rgb, image_a_rgb, torch.zeros(1).type(dtype_long), torch.zeros(1).type(
                dtype_long), torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long)


        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)


        num_attempts = 50000
        num_non_matches_per_match = 150
        if self.debug:
            num_attempts = 20
            num_non_matches_per_match = 1

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose, 
                                                                           image_b_depth_numpy, image_b_pose, 
                                                                           num_attempts=num_attempts, img_a_mask=np.asarray(image_a_mask))

        if uv_a is None:
            print "No matches this time"
            return "matches", image_a_rgb, image_b_rgb, torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long)

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
            print "masking non-matches"
            image_b_mask = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        else:
            print "not masking non-matches"
            image_b_mask = None
            
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b, num_non_matches_per_match=num_non_matches_per_match, img_b_mask=image_b_mask)

        if self.debug:
            # only want to bring in plotting code if in debug mode
            import correspondence_plotter

            # Just show all images 
            # self.debug_show_data(image_a_rgb, image_a_depth, image_b_pose,
            #                  image_b_rgb, image_b_depth, image_b_pose)
            uv_a_long = (torch.t(uv_a[0].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1), 
                     torch.t(uv_a[1].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1))
            uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )
            
            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy, uv_a, uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth_numpy, image_b_rgb, image_b_depth_numpy,
                                                  uv_a_long, uv_b_non_matches_long,
                                                  use_previous_plot=(fig,axes),
                                                  circ_color='r')


        if self.tensor_transform is not None:
            image_a_rgb, image_b_rgb = self.both_to_tensor([image_a_rgb, image_b_rgb])

        uv_a_long = (torch.t(uv_a[0].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1), 
                     torch.t(uv_a[1].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1))
        uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )

        # flatten correspondences and non_correspondences
        uv_a = uv_a[1].type(dtype_long)*640+uv_a[0].type(dtype_long)
        uv_b = uv_b[1].type(dtype_long)*640+uv_b[0].type(dtype_long)
        uv_a_long = uv_a_long[1].type(dtype_long)*640+uv_a_long[0].type(dtype_long)
        uv_b_non_matches_long = uv_b_non_matches_long[1].type(dtype_long)*640+uv_b_non_matches_long[0].type(dtype_long)
        uv_a_long = uv_a_long.squeeze(1)
        uv_b_non_matches_long = uv_b_non_matches_long.squeeze(1)

        return "matches", image_a_rgb, image_b_rgb, uv_a, uv_b, uv_a_long, uv_b_non_matches_long

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


    def different_enough(self, pose_1, pose_2):
        translation_1 = np.asarray(pose_1[0,3], pose_1[1,3], pose_1[2,3])
        translation_2 = np.asarray(pose_2[0,3], pose_2[1,3], pose_2[2,3])

        translation_threshold = 0.2 # meters
        if np.linalg.norm(translation_1 - translation_2) > translation_threshold:
            return True

        # later implement something that is different_enough for rotations?
        return False

    def get_random_rgb_image_filename(self, scene_directory):
        rgb_images_regex = os.path.join(scene_directory, "images/*_rgb.png")
        all_rgb_images_in_scene = sorted(glob.glob(rgb_images_regex))
        random_rgb_image = random.choice(all_rgb_images_in_scene)
        return random_rgb_image

    def get_specific_rgb_image_filname(self, scene_name, img_index):
        DeprecationWarning("use get_specific_rgb_image_filename instead")
        return self.get_specific_rgb_image_filename(scene_name, img_index)

    def get_specific_rgb_image_filename(self, scene_name, img_index):
        """
        Returns the filename for the specific RGB image
        :param scene_name:
        :param img_index: int or str
        :return:
        """
        if isinstance(img_index, int):
            img_index = utils.getPaddedString(img_index)

        scene_directory = self.get_full_path_for_scene(scene_name)
        images_dir = os.path.join(scene_directory, "images")
        rgb_image_filename = os.path.join(images_dir, img_index + "_rgb.png")
        return rgb_image_filename

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

    def get_depth_filename(self, rgb_image):
        prefix = rgb_image.split("rgb")[0]
        depth_filename = prefix+"depth.png"
        return depth_filename

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