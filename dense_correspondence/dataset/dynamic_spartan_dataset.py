from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType
from spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType

import os
import numpy as np
import logging
import glob
import random
import copy

import torch

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision
from torchvision import transforms

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics


import dense_correspondence_manipulation.utils.constants as constants


utils.add_dense_correspondence_to_python_path()
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation

from dense_correspondence.dataset.scene_structure import SceneStructure



class DynamicSpartanDataset(SpartanDataset):

    def __init__(self, debug=False, mode="train", config=None, config_expanded=None):
        """
        :param config: This is for creating a dataset from a composite dataset config file.
            This is of the form:

                logs_root_path: logs_proto # path relative to utils.get_data_dir()

                single_object_scenes_config_files:
                - caterpillar_17_scenes.yaml
                - baymax.yaml

                multi_object_scenes_config_files:
                - multi_object.yaml

        :type config: dict()

        :param config_expanded: When a config is read, it is parsed into an expanded form
            which is actually used as self._config.  See the function _setup_scene_data()
            for how this is done.  We want to save this expanded config to disk as it contains
            all config information.  If loading a previously-used dataset configuration, we want
            to pass in the config_expanded.
        :type config_expanded: dict()
        """
        SpartanDataset.__init__(self, debug=debug, mode=mode, config=config, config_expanded=config_expanded)
        # HACK
        self.num_images_total = 1000 

    def __len__(self):
        return 1000

    def get_camera_info_dict(self, scene_name, camera_num):
        scene_directory = self.get_full_path_for_scene(scene_name)
        camera_info_filename = os.path.join(scene_directory, "images_camera_"+str(camera_num), "camera_info.yaml")
        return utils.getDictFromYamlFilename(camera_info_filename)

    def get_pose_data(self, scene_name, camera_num):
        camera_info_dict = self.get_camera_info_dict(scene_name, camera_num)
        return camera_info_dict["extrinsics"]

    def get_K_matrix(self, scene_name, camera_num):
        camera_info_dict = self.get_camera_info_dict(scene_name, camera_num)
        K = camera_info_dict["camera_matrix"]["data"]
        return np.asarray(K).reshape(3,3)

    def get_pose_from_scene_camera_idx(self, scene_name, camera_num, idx):
        """
        :param scene_name: str
        :param img_idx: int
        :return: 4 x 4 numpy array
        """
        idx = int(idx)
        pose_data = self.get_pose_data(scene_name, camera_num)
        return utils.homogenous_transform_from_dict(pose_data)

    def get_random_rgbd_mask_pose(self):
        scene_name = self.get_random_scene_name()
        idx = self.get_random_image_index(scene_name)
        camera_num = random.choice([0,1])
        return self.get_rgbd_mask_pose(scene_name, camera_num, idx)

    def get_rgbd_mask_pose(self, scene_name, camera_num, img_idx):
        """
        Returns rgb image, depth image, mask and pose.
        :param scene_name:
        :type scene_name: str
        :param img_idx:
        :type img_idx: int
        :return: rgb, depth, mask, pose
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        rgb_file = self.get_image_filename(scene_name, camera_num, img_idx, ImageType.RGB)
        rgb = self.get_rgb_image(rgb_file)

        depth_file = self.get_image_filename(scene_name, camera_num, img_idx, ImageType.DEPTH)
        depth = self.get_depth_image(depth_file)

        mask_file = self.get_image_filename(scene_name, camera_num, img_idx, ImageType.MASK)
        mask = self.get_mask_image(mask_file)

        pose = self.get_pose_from_scene_camera_idx(scene_name, camera_num, img_idx)

        return rgb, depth, mask, pose


    def get_rgb_image_from_scene_name_and_idx_and_cam(self, scene_name, idx, camera_num):
        rgb_file = self.get_image_filename(scene_name, camera_num, idx, ImageType.RGB)
        return self.get_rgb_image(rgb_file)

    def get_mask_image_from_scene_name_and_idx_and_cam(self, scene_name, idx, camera_num):
        mask_file = self.get_image_filename(scene_name, camera_num, idx, ImageType.MASK)
        return self.get_mask_image(mask_file)

    def get_image_filename(self, scene_name, camera_num, img_index, image_type):
        """
        Get the image filename for that scene and image index
        :param scene_name: str
        :param img_index: str or int
        :param image_type: ImageType
        :return:
        """

        scene_directory = self.get_full_path_for_scene(scene_name)

        camera_dir = "images_camera_"+str(camera_num)
        os.path.join(scene_directory, camera_dir)

        if image_type == ImageType.RGB:
            images_dir = camera_dir
            file_extension = "_rgb.png"
        elif image_type == ImageType.DEPTH:
            images_dir = camera_dir
            file_extension = "_depth.png"
        elif image_type == ImageType.MASK:
            images_dir = os.path.join(camera_dir, 'image_masks')
            file_extension = "_mask.png"
        else:
            raise ValueError("unsupported image type")

        if isinstance(img_index, int):
            img_index = utils.getPaddedString(img_index, width=SpartanDataset.PADDED_STRING_WIDTH)

        scene_directory = self.get_full_path_for_scene(scene_name)
        if not os.path.isdir(scene_directory):
            raise ValueError("scene_name = %s doesn't exist" %(scene_name))

        return os.path.join(scene_directory, images_dir, img_index + file_extension)

    def get_random_camera_nums_for_image_index(self, idx):
        # HACK
        return np.random.permutation([0,1])

    def get_random_image_index(self, scene_name):
        scene_directory = self.get_full_path_for_scene(scene_name)
        state_info_filename = os.path.join(scene_directory, "states.yaml")
        state_info_dict = utils.getDictFromYamlFilename(state_info_filename)
        image_idxs = state_info_dict.keys() # list of integers
        return random.choice(image_idxs)

    def get_default_dynamic_dataset_K_matrix(self):
        K = np.zeros((3,3))
        K[0,0] = 471.09511257614645 # focal x
        K[1,1] = 471.09511257614645 # focal y
        K[0,2] = 319.5 # principal point x
        K[1,2] = 239.5 # principal point y
        K[2,2] = 1.0
        return K
        
    def get_within_scene_data(self, scene_name, metadata, for_synthetic_multi_object=False):
        """
        The method through which the dataset is accessed for training.

        Each call is is the result of
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

        5th, 6th return args: masked_non_matches_a, masked_non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        9th, 10th return args: blind_non_matches_a, blind_non_matches_b
        9th, 10th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        11th return arg: metadata useful for plotting, and-or other flags for loss functions
        11th rtype: dict

        Return values 3,4,5,6,7,8,9,10 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        If no datapoints were found for some type of match or non-match then we return
        our "special" empty tensor. Note that due to the way the pytorch data loader
        functions you cannot return an empty tensor like torch.FloatTensor([]). So we
        return SpartanDataset.empty_tensor()

        """

        SD = SpartanDataset

        idx = self.get_random_image_index(scene_name)
        camera_num_a, camera_num_b = self.get_random_camera_nums_for_image_index(idx)

        metadata['image_a_idx'] = idx
        metadata["camera_a_num"] = camera_num_a
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, camera_num_a, idx)

        metadata['image_b_idx'] = idx
        metadata["camera_b_num"] = camera_num_b
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, camera_num_b, idx)
        


        # THE REST IS THE SAME BUT I JUST NEEED A REFACTOR


        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        if self.sample_matches_only_off_mask:
            correspondence_mask = np.asarray(image_a_mask)
        else:
            correspondence_mask = None

        #K = self.get_default_dynamic_dataset_K_matrix()
        K_a = self.get_K_matrix(scene_name, camera_num_a)
        K_b = self.get_K_matrix(scene_name, camera_num_b)

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose,
                                                                            image_b_depth_numpy, image_b_pose,
                                                                            img_a_mask=correspondence_mask,
                                                                            num_attempts=self.num_matching_attempts,
                                                                            K_a=K_a,
                                                                            K_b=K_b)

        if for_synthetic_multi_object:
            return image_a_rgb, image_b_rgb, image_a_depth, image_b_depth, image_a_mask, image_b_mask, uv_a, uv_b


        if uv_a is None:
            logging.info("no matches found, returning")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)


        # data augmentation
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)

        if not self.debug:
            [image_a_rgb, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], uv_a)
            [image_b_rgb, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_mask], uv_b)
        else:  # also mutate depth just for plotting
            [image_a_rgb, image_a_depth, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation(
                [image_a_rgb, image_a_depth, image_a_mask], uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_depth, image_b_mask], uv_b)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)


        # find non_correspondences
        image_b_mask_torch = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        uv_b_masked_non_matches = \
            correspondence_finder.create_non_correspondences(uv_b,
                                                             image_b_shape,
                                                             num_non_matches_per_match=self.num_masked_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_torch)


        if self._use_image_b_mask_inv:
            image_b_mask_inv = 1 - image_b_mask_torch
        else:
            image_b_mask_inv = None

        uv_b_background_non_matches = correspondence_finder.create_non_correspondences(uv_b,
                                                                            image_b_shape,
                                                                            num_non_matches_per_match=self.num_background_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_inv)



        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        matches_a = SD.flatten_uv_tensor(uv_a, image_width)
        matches_b = SD.flatten_uv_tensor(uv_b, image_width)

        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(uv_a, uv_b_masked_non_matches, self.num_masked_non_matches_per_match)

        masked_non_matches_a = SD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)


        # Non-masked non-matches
        uv_a_background_long, uv_b_background_non_matches_long = self.create_non_matches(uv_a, uv_b_background_non_matches,
                                                                            self.num_background_non_matches_per_match)

        background_non_matches_a = SD.flatten_uv_tensor(uv_a_background_long, image_width).squeeze(1)
        background_non_matches_b = SD.flatten_uv_tensor(uv_b_background_non_matches_long, image_width).squeeze(1)


        # make blind non matches
        matches_a_mask = SD.mask_image_from_uv_flat_tensor(matches_a, image_width, image_height)
        image_a_mask_torch = torch.from_numpy(np.asarray(image_a_mask)).long()
        mask_a_flat = image_a_mask_torch.view(-1,1).squeeze(1)
        blind_non_matches_a = (mask_a_flat - matches_a_mask).nonzero()

        no_blind_matches_found = False
        if len(blind_non_matches_a) == 0:
            no_blind_matches_found = True
        else:

            blind_non_matches_a = blind_non_matches_a.squeeze(1)
            num_blind_samples = blind_non_matches_a.size()[0]

            if num_blind_samples > 0:
                # blind_uv_b is a tuple of torch.LongTensor
                # make sure we check that blind_uv_b is not None and that it is non-empty


                blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(image_b_mask_torch, num_blind_samples)

                if blind_uv_b[0] is None:
                    no_blind_matches_found = True
                elif len(blind_uv_b[0]) == 0:
                    no_blind_matches_found = True
                else:
                    blind_non_matches_b = utils.uv_to_flattened_pixel_locations(blind_uv_b, image_width)

                    if len(blind_non_matches_b) == 0:
                        no_blind_matches_found = True
            else:
                no_blind_matches_found = True

        if no_blind_matches_found:
            blind_non_matches_a = blind_non_matches_b = SD.empty_tensor()


        if self.debug:
            # downsample so can plot
            num_matches_to_plot = 10
            plot_uv_a, plot_uv_b = SD.subsample_tuple_pair(uv_a, uv_b, num_samples=num_matches_to_plot)

            plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = SD.subsample_tuple_pair(uv_a_masked_long, uv_b_masked_non_matches_long, num_samples=num_matches_to_plot*3)

            plot_uv_a_background_long, plot_uv_b_background_non_matches_long = SD.subsample_tuple_pair(uv_a_background_long, uv_b_background_non_matches_long, num_samples=num_matches_to_plot*3)

            blind_uv_a = utils.flattened_pixel_locations_to_u_v(blind_non_matches_a, image_width)
            plot_blind_uv_a, plot_blind_uv_b = SD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot*10)


        if self.debug:
            # only want to bring in plotting code if in debug mode
            import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter

            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='r')

                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_background_long, plot_uv_b_background_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='b')


                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_blind_uv_a, plot_blind_uv_b,
                                                                   circ_color='k', show=True)

                # Mask-plotting city
                import matplotlib.pyplot as plt
                plt.imshow(np.asarray(image_a_mask))
                plt.title("Mask of img a object pixels")
                plt.show()

                plt.imshow(np.asarray(image_a_mask) - 1)
                plt.title("Mask of img a background")
                plt.show()

                temp = matches_a_mask.view(image_height, -1)
                plt.imshow(temp)
                plt.title("Mask of img a object pixels for which there was a match")
                plt.show()

                temp2 = (mask_a_flat - matches_a_mask).view(image_height, -1)
                plt.imshow(temp2)
                plt.title("Mask of img a object pixels for which there was NO match")
                plt.show()



        depth_to_tensor = transforms.Compose([transforms.ToTensor()])
        depth_a_torch = depth_to_tensor(image_a_depth).float() / 1000.0
        depth_b_torch = depth_to_tensor(image_b_depth).float() / 1000.0
        return metadata["type"], image_a_rgb, image_b_rgb, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, metadata, depth_a_torch, depth_b_torch
