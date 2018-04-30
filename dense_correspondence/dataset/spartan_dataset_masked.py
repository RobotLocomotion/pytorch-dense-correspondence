from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType

import os
import numpy as np
import logging
import glob
import random

import torch
import torch.utils.data as data
from torchvision import transforms


import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics
from torchvision import transforms
import dense_correspondence_manipulation.utils.constants as constants
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation


class SpartanDatasetDataType:
    SINGLE_OBJECT_WITHIN_SCENE = 0

class SpartanDataset(DenseCorrespondenceDataset):

    PADDED_STRING_WIDTH = 6

    def __init__(self, debug=False, mode="train", config=None):

        assert config is not None
        DenseCorrespondenceDataset.__init__(self, debug=debug)

        self._config = config
        self._setup_scene_data()

        self.num_matching_attempts = 50
        self.num_non_matches_per_match = 150

        #
        #
        #
        # if config is None:
        #     dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
        #                                'dataset',
        #                                'spartan_dataset_masked.yaml')
        #
        #     dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)
        #     self._config = dataset_config
        #
        #     self.set_train_test_split_from_yaml(self._config)
        # else:
        #     # assume config has already been parsed
        #     self._config = config
        #     self.logs_root_path = utils.convert_to_absolute_path(self._config['logs_root_path'])
        #     self.set_train_test_split_from_yaml(self._config)


        self._pose_data = dict()

        self._initialize_rgb_image_to_tensor()

        
        if mode == "test":
            self.set_test_mode()

        self.init_length()
        print "Using SpartanDataset:"
        print "   - in", self.mode, "mode"
        print "   - total images:    ", self.num_images_total



    def _setup_scene_data(self):
        """
        Initializes the data for all the different types of scenes

        Creates two class attributes

        self._single_object_scene_dict

        Each entry of self._single_object_scene_dict is a dict with keys {"test", "train"}. The
        values are lists of scenes

        self._multi_object_scene_dict

        self._single_object_scene_dict has (key, value) = (object_id, scene config for that object)

        self._multi_object_scene_dict has (key, value) = ("train"/"test", list of scenes)
        Note that the scenes have absolute paths here



        :return:
        :rtype:
        """

        self.logs_root_path = utils.convert_to_absolute_path(self._config['logs_root_path'])

        self._single_object_scene_dict = dict()

        prefix = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset')

        for config_file in self.config["single_object_scenes_config_files"]:
            config_file = os.path.join(prefix, config_file)
            single_object_scene_config = utils.getDictFromYamlFilename(config_file)
            object_id = single_object_scene_config["object_id"]
            self._single_object_scene_dict[object_id] = single_object_scene_config

        # will have test and train entries
        # each one is a list of scenes
        self._multi_object_scene_dict = {"train": [], "test": []}


        for config_file in self.config["multi_object_scenes_config_files"]:
            config_file = os.path.join(prefix, config_file)
            multi_object_scene_config = utils.getDictFromYamlFilename(config_file)

            for key, scene_list in self._multi_object_scene_dict.iteritems():
                for scene_name in multi_object_scene_config[key]:
                    scene_list.append(scene_name)

    def scene_generator(self, mode=None):
        """
        Returns an generator that traverses all the scenes
        :return:
        :rtype:
        """
        if mode is None:
            mode = self.mode

        for object_id, single_object_scene_dict in self._single_object_scene_dict.iteritems():
            for scene_name in single_object_scene_dict[mode]:
                yield scene_name

        for scene_name in self._multi_object_scene_dict[mode]:
            yield scene_name

    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

    def get_pose(self, rgb_filename):
        scene_directory = rgb_filename.split("images")[0]
        index = self.get_index(rgb_filename)
        pose_list = self.get_pose_list(scene_directory, "images.posegraph")
        pose_elasticfusion = self.get_pose_from_list(int(index), pose_list)
        pose_matrix4 = self.elasticfusion_pose_to_homogeneous_transform(pose_elasticfusion)
        return pose_matrix4

    def get_full_path_for_scene(self, scene_name, ):
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

        for scene_name in self.scene_generator():
            self.get_pose_data(scene_name)

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

    def get_random_object_id(self):
        """
        Returns a random object_id
        :return:
        :rtype:
        """
        object_id_list = self._single_object_scene_dict.keys()
        return random.choice(object_id_list)

    def get_random_single_object_scene_name(self, object_id):
        """
        Returns a random scene name for that object
        :param object_id: str
        :type object_id:
        :return: str
        :rtype:
        """

        scene_list = self._single_object_scene_dict[object_id][self.mode]
        return random.choice(scene_list)


    def get_random_multi_object_scene_name(self):
        """
        Returns a random multi object scene name
        :return:
        :rtype:
        """
        return random.choice(self._multi_object_scene_dict[self.mode])


    def get_number_of_unique_single_objects(self):
        """
        Returns the number of unique objects in this dataset with single object scenes
        :return:
        :rtype:
        """
        return len(self._single_object_scene_dict.keys())

    def has_multi_object_scenes(self):
        """
        Returns true if there are multi-object scenes in this datase
        :return:
        :rtype:
        """
        return len(self._multi_object_scene_dict["train"]) > 0


    def get_single_object_within_scene_data(self):
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

        Return values 3,4,5,6,7,8 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        """



        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")

        # stores metadata about this data
        metadata = dict()
        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)

        metadata["object_id"] = object_id
        metadata["scene_name"] = scene_name
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE


        image_a_idx = self.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)

        metadata['image_a_idx'] = image_a_idx

        # image b
        image_b_idx = self.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)
        metadata['image_b_idx'] = image_b_idx
        if image_b_idx is None:
            logging.info("no frame with sufficiently different pose found, returning")
            # TODO: return something cleaner than no-data
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)

        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose,
                                                                            image_b_depth_numpy, image_b_pose,
                                                                            num_attempts=self.num_matching_attempts,
                                                                            img_a_mask=np.asarray(image_a_mask))

        # find non_correspondences
        num_masked_non_matches_per_match = self.num_non_matches_per_match
        num_non_masked_non_matches_per_match = self.num_non_matches_per_match

        if uv_a is None:
            logging.info("no matches found, returning")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)

        if self.debug:
            # downsample so can plot
            num_matches_to_plot = 10
            num_non_masked_non_matches_per_match = 10
            num_masked_non_matches_per_match = 1
            indexes_to_keep = (torch.rand(num_matches_to_plot) * len(uv_a[0])).floor().type(torch.LongTensor)
            uv_a = (torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep))
            uv_b = (torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep))

        # data augmentation
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)


        if not self.debug:
            [image_a_rgb], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb], uv_a)
            [image_b_rgb, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_mask], uv_b)
        else:  # also mutate depth just for plotting
            [image_a_rgb, image_a_depth], uv_a = correspondence_augmentation.random_image_and_indices_mutation(
                [image_a_rgb, image_a_depth], uv_a)
            [image_b_rgb, image_b_depth,
             image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_depth, image_b_mask], uv_b)
            image_a_depth_numpy = np.asarray(image_a_depth)
            image_b_depth_numpy = np.asarray(image_b_depth)


        # find non_correspondences


        image_b_mask = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)

        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[1]

        uv_b_masked_non_matches = \
            correspondence_finder.create_non_correspondences(uv_b,
                                                             image_b_shape,
                                                             num_non_matches_per_match=num_masked_non_matches_per_match,
                                                                            img_b_mask=image_b_mask)


        image_b_mask_inv = 1 - image_b_mask

        uv_b_non_masked_non_matches = correspondence_finder.create_non_correspondences(uv_b,
                                                                            image_b_shape,
                                                                            num_non_matches_per_match=num_non_masked_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_inv)



        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        def flatten_uv_tensor(uv_tensor):
            """
            Flattens a uv_tensor to single dimensional tensor
            :param uv_tensor:
            :type uv_tensor:
            :return:
            :rtype:
            """
            return uv_tensor[1].long() * image_width + uv_tensor[0].long()

        matches_a = flatten_uv_tensor(uv_a)
        matches_b = flatten_uv_tensor(uv_b)


        def create_non_matches(uv_a, uv_b_non_matches, multiplier):
            """
            Simple inline wrapper for repeated code
            :param uv_a:
            :type uv_a:
            :param uv_b_non_matches:
            :type uv_b_non_matches:
            :param multiplier:
            :type multiplier:
            :return:
            :rtype:
            """
            uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                         torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

            uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

            return uv_a_long, uv_b_non_matches_long




        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = create_non_matches(uv_a, uv_b_masked_non_matches, num_masked_non_matches_per_match)

        masked_non_matches_a = flatten_uv_tensor(uv_a_masked_long)
        masked_non_matches_a.squeeze(1)

        masked_non_matches_b = flatten_uv_tensor(uv_b_masked_non_matches_long)
        masked_non_matches_b.squeeze(1)



        # Non-masked non-matches
        uv_a_non_masked_long, uv_b_non_masked_non_matches_long = create_non_matches(uv_a, uv_b_non_masked_non_matches,
                                                                            num_non_masked_non_matches_per_match)

        non_masked_non_matches_a = flatten_uv_tensor(uv_a_non_masked_long)
        non_masked_non_matches_a.squeeze(1)

        non_masked_non_matches_b = flatten_uv_tensor(uv_b_non_masked_non_matches_long)
        non_masked_non_matches_b.squeeze(1)

        if self.debug:
            # only want to bring in plotting code if in debug mode
            import correspondence_plotter

            # Just show all images

            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy, uv_a,
                                                                               uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL,
                                                                   image_b_depth_numpy,
                                                                   uv_a_masked_long, uv_b_masked_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='r')

                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               uv_a,
                                                                               uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, image_b_rgb_PIL,
                                                                   image_b_depth_numpy,
                                                                   uv_a_non_masked_long, uv_b_non_masked_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='b')



        return "matches", image_a_rgb, image_b_rgb, matches_a, matches_b, masked_non_matches_a, masked_non_matches_a, non_masked_non_matches_a, non_masked_non_matches_b, metadata


    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_MEAN

        # return self.config["image_normalization"]["mean"]

        
        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_STD_DEV

        # return self.config["image_normalization"]["std_dev"]

        return constants.DEFAULT_IMAGE_STD_DEV

        

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """

        return self._rgb_image_to_tensor(img)

    @property
    def config(self):
        return self._config

    @staticmethod
    def make_default_10_scenes_drill():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                   'dataset',
                                   '10_drill_scenes.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset

    @staticmethod
    def make_default_caterpillar():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                   'dataset',
                                   'caterpillar_17_scenes.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset






