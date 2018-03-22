import torch
import torch.utils.data as data

import os
import math
import yaml
import numpy as np
import random
import glob
from PIL import Image

# For debuggig only
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../../pytorch-segmentation-detection/vision/')
from torchvision import transforms
sys.path.append('../../pytorch-segmentation-detection/')
from pytorch_segmentation_detection.transforms import ComposeJoint
sys.path.append('../correspondence_tools/')
import correspondence_finder
import correspondence_plotter

# This implements an abstract Dataset class in PyTorch
# to load in LabelFusion data (labelfusion.csail.mit.edu)
#
# in particular note:
# __len__     is overloaded
# __getitem__ is overloaded
#
# For more info see:
# http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset

class DenseCorrespondenceDataset(data.Dataset):

    def __init__(self, debug=False):
        
        self.debug = debug
        
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
        scene_directory = self.get_random_scene_directory()

        # image a
        image_a_rgb, image_a_depth, image_a_pose, image_a_mask = self.get_random_rgbd_with_pose_and_mask(scene_directory)
        
        # image b
        image_b_rgb, image_b_depth, image_b_pose = self.get_different_rgbd_with_pose(scene_directory, image_a_pose)


        num_attempts = 50000
        num_non_matches_per_match = 150
        if self.debug:
            num_attempts = 20
            num_non_matches_per_match = 1

        # find correspondences    
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth, image_a_pose, 
                                                                           image_b_depth, image_b_pose, 
                                                                           num_attempts=num_attempts, img_a_mask=image_a_mask)


        # find non_correspondences
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_a, uv_b, num_non_matches_per_match=num_non_matches_per_match)

        if self.debug:

            # Just show all images 
            # self.debug_show_data(image_a_rgb, image_a_depth, image_b_pose,
            #                  image_b_rgb, image_b_depth, image_b_pose)
            uv_a_long = (torch.t(uv_a[0].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1), 
                     torch.t(uv_a[1].repeat(num_non_matches_per_match, 1)).contiguous().view(-1,1))
            uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )
            
            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth, image_b_rgb, image_b_depth, uv_a, uv_b, show=False)
                correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth, image_b_rgb, image_b_depth,
                                                  uv_a_long, uv_b_non_matches_long,
                                                  use_previous_plot=(fig,axes),
                                                  circ_color='r')


        if self.tensor_transform is not None:
            image_a_rgb, image_b_rgb = self.both_to_tensor([image_a_rgb, image_b_rgb])


        if uv_a is None:
            print "No matches this time"
            return "matches", image_a_rgb, image_b_rgb, torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long), torch.zeros(1).type(dtype_long)

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

    def get_random_rgbd_with_pose(self, scene_directory):
        rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
        depth_filename = self.get_depth_filename(rgb_filename) 

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        pose  = self.get_pose(rgb_filename)

        return rgb, depth, pose

    def get_random_rgbd_with_pose_and_mask(self, scene_directory):
        rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
        depth_filename = self.get_depth_filename(rgb_filename)
        mask_filename  = self.get_mask_filename(rgb_filename)

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        pose  = self.get_pose(rgb_filename)
        mask  = self.get_mask_image(mask_filename)

        return rgb, depth, pose, mask

    def get_random_rgb_with_mask(self, scene_directory):
        rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
        mask_filename  = self.get_mask_filename(rgb_filename)

        rgb   = self.get_rgb_image(rgb_filename)
        mask  = self.get_mask_image(mask_filename)

        return rgb, mask

    def get_different_rgbd_with_pose(self, scene_directory, image_a_pose):
        # try to get a far-enough-away pose
        # if can't, then just return last sampled pose
        num_attempts = 0
        while num_attempts < 10:
            rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
            depth_filename = self.get_depth_filename(rgb_filename)
            pose           = self.get_pose(rgb_filename) 
            if self.different_enough(image_a_pose, pose):
                break
            num_attempts += 1

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        return rgb, depth, pose

    def get_rgb_image(self, rgb_filename):
        return Image.open(rgb_filename).convert('RGB')

    def get_depth_image(self, depth_filename):
        return np.asarray(Image.open(depth_filename))

    def get_mask_image(self, mask_filename):
        return np.asarray(Image.open(mask_filename))

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

    def get_specific_rgbd_with_pose(self, scene_name, img_index):
        rgb_filename   = self.get_specific_rgb_image_filname(scene_name, img_index)
        depth_filename = self.get_depth_filename(rgb_filename) 

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        pose  = self.get_pose(rgb_filename)

        return rgb, depth, pose

    def get_specific_rgb_image_filname(self, scene_name, img_index):
        scene_directory = self.get_full_path_for_scene(scene_name)
        images_dir = os.path.join(scene_directory, "images")
        rgb_image_filename = os.path.join(images_dir, img_index+"_rgb.png")
        return rgb_image_filename

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
        full_path = os.path.join(self.logs_root_path, scene_name)
        return full_path

    def get_random_scene_directory(self):
        scene_name = random.choice(self.scenes)
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

        relative_path = config_dict[key]
        full_path = os.path.join(os.environ['HOME'], relative_path)
        return full_path

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