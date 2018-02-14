import torch
import torch.utils.data as data

import os
import math
import numpy as np
import random
import glob
from PIL import Image

# For debuggig only
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '../pytorch-segmentation-detection/vision/')
from torchvision import transforms
sys.path.append('../pytorch-segmentation-detection/')
from pytorch_segmentation_detection.transforms import ComposeJoint
sys.path.append('../')
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

class LabelFusionDataset(data.Dataset):

    def __init__(self, debug=False):
        
        self.debug = debug

        self.labelfusion_logs_test_root_path = "/media/peteflo/3TBbackup/local-only/logs_test/"
        
        # later this could just automatically populate all scenes available
        # for now though, need a list since haven't extracted all depths
        self.scenes = ["2017-06-16-21",
                       "2017-06-14-63",
                       "2017-06-13-12"]

        self.init_length()

        self.tensor_transform = ComposeJoint(
                [
                    [transforms.ToTensor(), None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(x).long()) ]
                ])

        # Pete Todo: would be great to automate this later        
        # if download:
            
        #     self._download_dataset()
        #     self._extract_dataset()
        #     self._prepare_dataset()
        
        # Pete Todo: separate train/val?
        # if train:
        #     self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[0]
        # else:
        #     self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[1]
                   
        
    def __len__(self):
        return self.num_images_total
    
    def __getitem__(self, index):

        # pick a scene
        scene_directory = self.get_random_scene_directory()

        # image a
        image_a_rgb, image_a_depth, image_a_pose = self.get_random_rgbd_with_pose(scene_directory)
        
        # image b
        image_b_rgb, image_b_depth, image_b_pose = self.get_different_rgbd_with_pose(scene_directory, image_a_pose)

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth, image_a_pose, 
                                                                            image_b_depth, image_b_pose, 
                                                                            num_attempts=5000)

        # find non_correspondences
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_a, uv_b, num_non_matches_per_match=100)
        
        if self.debug:

            # Just show all images 
            # self.debug_show_data(image_a_rgb, image_a_depth, image_b_pose,
            #                  image_b_rgb, image_b_depth, image_b_pose)
            
            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth, image_b_rgb, image_b_depth, uv_a, uv_b, show=False)

                uv_a_long = (torch.t(uv_a[0].repeat(3, 1)).contiguous().view(-1,1), torch.t(uv_a[1].repeat(3, 1)).contiguous().view(-1,1))
                uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )
                correspondence_plotter.plot_correspondences_direct(image_a_rgb, image_a_depth, image_b_rgb, image_b_depth,
                                                  uv_a_long, uv_b_non_matches_long,
                                                  use_previous_plot=(fig,axes),
                                                  circ_color='r')


        if self.tensor_transform is not None:
            rgbd_a = self.tensor_transform([image_a_rgb, image_a_depth])
            rgbd_a.append(image_a_pose)
            rgbd_b = self.tensor_transform([image_b_rgb, image_b_depth])
            rgbd_a.append(image_b_pose)

        return rgbd_a, rgbd_b

    def get_random_rgbd_with_pose(self, scene_directory):
        rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
        depth_filename = self.get_depth_filename(rgb_filename)
        time_filename  = self.get_time_filename(rgb_filename) 

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        pose  = self.get_pose(time_filename)

        return rgb, depth, pose

    def get_different_rgbd_with_pose(self, scene_directory, image_a_pose):
        # try to get a far-enough-away pose
        # if can't, then just return last sampled pose
        num_attempts = 0
        while num_attempts < 10:
            rgb_filename   = self.get_random_rgb_image_filename(scene_directory)
            depth_filename = self.get_depth_filename(rgb_filename)
            time_filename  = self.get_time_filename(rgb_filename)
            pose           = self.get_pose(time_filename) 
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
        time_filename  = self.get_time_filename(rgb_filename) 

        rgb   = self.get_rgb_image(rgb_filename)
        depth = self.get_depth_image(depth_filename)
        pose  = self.get_pose(time_filename)

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

    def get_time_filename(self, rgb_image):
        prefix = rgb_image.split("rgb")[0]
        time_filename = prefix+"utime.txt"
        return time_filename

    # will happily do a more efficient way of grabbing pose
    # if this appears to be a bottleneck
    def get_pose(self, time_filename):
        time = self.get_time(time_filename)
        scene_directory = time_filename.split("images")[0]
        pose_list = self.get_pose_list(scene_directory)
        pose_labelfusion = self.get_pose_from_list(time, pose_list)
        pose_matrix4 = self.labelfusion_pose_to_homogeneous_transform(pose_labelfusion)
        return pose_matrix4

    def get_time(self, time_filename):
        with open (time_filename) as f:
            content = f.readlines()
        return int(content[0])/1e6

    def get_pose_list(self, scene_directory):
        posegraph_filename = os.path.join(scene_directory, "posegraph.posegraph")
        with open(posegraph_filename) as f:
            content = f.readlines()
        pose_list = [x.strip().split() for x in content]
        return pose_list

    def get_pose_from_list(self, time, pose_list):
        if (time <= float(pose_list[0][0])):
            pose = pose_list[0]
            pose = [float(x) for x in pose[1:]]
            return pose
        for pose in pose_list:
            if (time <= float(pose[0])):
                pose = [float(x) for x in pose[1:]]
                return pose
        print "did not find matching pose"
        quit()

    def labelfusion_pose_to_homogeneous_transform(self, lf_pose):
        homogeneous_transform = self.quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0,3] = lf_pose[0]
        homogeneous_transform[1,3] = lf_pose[1]
        homogeneous_transform[2,3] = lf_pose[2]
        return homogeneous_transform

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

    def get_full_path_for_scene(self, scene_name):
        full_path = os.path.join(self.labelfusion_logs_test_root_path, scene_name)
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