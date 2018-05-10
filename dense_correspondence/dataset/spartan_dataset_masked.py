from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType

import os
import numpy as np
import logging
import glob
import random

import torch

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision
from torchvision import transforms

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

import dense_correspondence_manipulation.utils.constants as constants
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation


class SpartanDatasetDataType:
    SINGLE_OBJECT_WITHIN_SCENE = 0
    SINGLE_OBJECT_ACROSS_SCENE = 1
    DIFFERENT_OBJECT = 2
    MULTI_OBJECT = 3


class SpartanDataset(DenseCorrespondenceDataset):

    PADDED_STRING_WIDTH = 6

    def __init__(self, debug=False, mode="train", config=None, config_expanded=None):
        """
        :param config: This is for creating a dataset from a composite dataset config file.
            This is of the form:

                logs_root_path: code/data_volume/pdc/logs_proto

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

        DenseCorrespondenceDataset.__init__(self, debug=debug)

        # Otherwise, all of these parameters should be set in
        # set_parameters_from_training_config()
        # which is called from training.py
        # and parameters are populated in training.yaml
        if self.debug:
            # NOTE: these are not the same as the numbers 
            # that get plotted in debug mode.
            # This is just so the dataset will "run".
            self._domain_randomize = False
            self.num_masked_non_matches_per_match = 5
            self.num_background_non_matches_per_match = 5
            self.cross_scene_num_samples = 1000
            self._use_image_b_mask_inv = True

        if config is not None:
            self._setup_scene_data(config)
        elif config_expanded is not None:
            self._parse_expanded_config(config_expanded)
        else:
            raise ValueError("You need to give me either a config or config_expanded")

        self._pose_data = dict()
        self._initialize_rgb_image_to_tensor()

        if mode == "test":
            self.set_test_mode()
        elif mode == "train":
            self.set_train_mode()
        else:
            raise ValueError("mode should be one of [test, train]")

        self.init_length()
        print "Using SpartanDataset:"
        print "   - in", self.mode, "mode"
        print "   - number of scenes", self._num_scenes
        print "   - total images:    ", self.num_images_total


    def __getitem__(self, index):
        """
        This overloads __getitem__ and is what is actually returned 
        using a torch dataloader.

        This small function randomly chooses one of our different 
        img pair types, then returns that type of data.
        """


        data_load_type = self._get_data_load_type()

        # Case 0: Same scene, same object
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
            print "Same scene, same object"
            return self.get_single_object_within_scene_data()

        # Case 1: Same object, different scene
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
            print "Same object, different scene"
            return self.get_single_object_across_scene_data()

        # Case 2: Different object
        if data_load_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
            print "Different object"
            return self.get_different_object_data()

        # Case 3: Multi object
        if data_load_type == SpartanDatasetDataType.MULTI_OBJECT:
            print "Multi object"
            return self.get_multi_object_within_scene_data()


    def _setup_scene_data(self, config):
        """
        Initializes the data for all the different types of scenes

        Creates two class attributes

        self._single_object_scene_dict

        Each entry of self._single_object_scene_dict is a dict with keys {"test", "train"}. The
        values are lists of scenes

        self._single_object_scene_dict has (key, value) = (object_id, scene config for that object)

        self._multi_object_scene_dict has (key, value) = ("train"/"test", list of scenes)

        Note that the scenes have absolute paths here
        """

        self.logs_root_path = utils.convert_to_absolute_path(config['logs_root_path'])

        self._single_object_scene_dict = dict()

        prefix = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset')

        for config_file in config["single_object_scenes_config_files"]:
            config_file = os.path.join(prefix, 'single_object', config_file)
            single_object_scene_config = utils.getDictFromYamlFilename(config_file)
            object_id = single_object_scene_config["object_id"]
            self._single_object_scene_dict[object_id] = single_object_scene_config

        # will have test and train entries
        # each one is a list of scenes
        self._multi_object_scene_dict = {"train": [], "test": [], "evaluation_labeled_data_path": []}

        for config_file in config["multi_object_scenes_config_files"]:
            config_file = os.path.join(prefix, 'multi_object', config_file)
            multi_object_scene_config = utils.getDictFromYamlFilename(config_file)

            for key, val in self._multi_object_scene_dict.iteritems():
                for item in multi_object_scene_config[key]:
                    val.append(item)

        self._config = dict()
        self._config["logs_root_path"] = config['logs_root_path']
        self._config["single_object"] = self._single_object_scene_dict
        self._config["multi_object"] = self._multi_object_scene_dict

        self._setup_data_load_types()

    def _parse_expanded_config(self, config_expanded):
        """
        If we have previously saved to disk a dict with:
        "single_object", and
        "multi_object" keys, 
        then we want to recreate a config from these.
        """
        self._config = config_expanded
        self._single_object_scene_dict = self._config["single_object"]
        self._multi_object_scene_dict = self._config["multi_object"] 
        self.logs_root_path =  utils.convert_to_absolute_path(self._config["logs_root_path"])

    def _setup_data_load_types(self):

        self._data_load_types = []
        if self.debug:
            self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE)
            # self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE)
            self._data_load_types.append(SpartanDatasetDataType.DIFFERENT_OBJECT)
            # self._data_load_types.append(SpartanDatasetDataType.MULTI_OBJECT)

    def _get_data_load_type(self):
        """
        Gets a random data load type from the allowable types
        :return: SpartanDatasetDataType
        :rtype:
        """
        return random.choice(self._data_load_types)

    def scene_generator(self, mode=None):
        """
        Returns a generator that traverses all the scenes
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
        """
        Checks if have not already loaded the pose_data.yaml for this scene,
        if haven't then loads it. Then returns the dict of the pose_data.yaml.
        :type scene_name: str
        :return: a dict() of the pose_data.yaml for the scene.
        :rtype: dict()
        """
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

    def get_random_object_id_and_int(self):
        """
        Returns a random object_id (a string) and its "int" (i.e. numerical unique id)
        :return:
        :rtype:
        """
        object_id_list = self._single_object_scene_dict.keys()
        random_object_id = random.choice(object_id_list)
        object_id_int = sorted(self._single_object_scene_dict.keys()).index(random_object_id)
        return random_object_id, object_id_int
        
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

    def get_different_scene_for_object(self, object_id, scene_name):
        """
        Return a different scene name
        :param object_id:
        :type object_id:
        :return:
        :rtype:
        """

        scene_list = self._single_object_scene_dict[object_id][self.mode]
        if len(scene_list) == 1:
            raise ValueError("There is only one scene of this object, can't sample a different one")

        idx_array = np.arange(0, len(scene_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)

        for idx in rand_idxs:
            scene_name_b = scene_list[idx]
            if scene_name != scene_name_b:
                return scene_name_b

        raise ValueError("It (should) be impossible to get here!!!!")

    def get_two_different_object_ids(self):
        """
        Returns two different random object ids
        :return: two object ids
        :rtype: two strings separated by commas
        """

        object_id_list = self._single_object_scene_dict.keys()
        if len(object_id_list) == 1:
            raise ValueError("There is only one object, can't sample a different one")

        idx_array = np.arange(0, len(object_id_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)

        object_1_id = object_id_list[rand_idxs[0]]
        object_2_id = object_id_list[rand_idxs[1]]

        assert object_1_id != object_2_id
        return object_1_id, object_2_id

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

    def get_random_scene_name(self):
        """
        Gets a random scene name across both single and multi object
        """
        types = []
        if self.has_multi_object_scenes():
            for _ in range(len(self._multi_object_scene_dict[self.mode])):
                types.append("multi")
        if self.get_number_of_unique_single_objects() > 0:
            for _ in range(self.get_number_of_unique_single_objects()):
                types.append("single")

        if len(types) == 0:
            raise ValueError("I don't think you have any scenes?")
        scene_type = random.choice(types)

        if scene_type == "multi":
            return self.get_random_multi_object_scene_name()
        if scene_type == "single":
            object_id = self.get_random_object_id()
            return self.get_random_single_object_scene_name(object_id)

    def get_single_object_within_scene_data(self):
        """
        Simple wrapper around get_within_scene_data(), for the single object case
        """
        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")

        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)

        metadata = dict()
        metadata["object_id"] = object_id
        metadata["scene_name"] = scene_name
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE

        return self.get_within_scene_data(scene_name, metadata)

    def get_multi_object_within_scene_data(self):
        """
        Simple wrapper around get_within_scene_data(), for the multi object case
        """

        if not self.has_multi_object_scenes():
            raise ValueError("There are no multi object scenes in this dataset")

        scene_name = self.get_random_multi_object_scene_name()

        metadata = dict()
        metadata["scene_name"] = scene_name
        metadata["type"] = SpartanDatasetDataType.MULTI_OBJECT

        return self.get_within_scene_data(scene_name, metadata)


    def get_within_scene_data(self, scene_name, metadata):
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
                                                                            img_a_mask=np.asarray(image_a_mask))


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
        uv_a_masked_long, uv_b_masked_non_matches_long = create_non_matches(uv_a, uv_b_masked_non_matches, self.num_masked_non_matches_per_match)

        masked_non_matches_a = SD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)


        # Non-masked non-matches
        uv_a_background_long, uv_b_background_non_matches_long = create_non_matches(uv_a, uv_b_background_non_matches,
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
            import correspondence_plotter

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



        return metadata["type"], image_a_rgb, image_b_rgb, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b, metadata

    def get_single_object_across_scene_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the single object case
        """
        metadata = dict()
        object_id = self.get_random_object_id()
        scene_name_a = self.get_random_single_object_scene_name(object_id)
        scene_name_b = self.get_different_scene_for_object(object_id, scene_name_a)
        metadata["object_id"] = object_id
        metadata["scene_name_a"] = scene_name_a
        metadata["scene_name_b"] = scene_name_b
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)

    def get_different_object_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the different object case
        """
        metadata = dict()
        object_id_a, object_id_b = self.get_two_different_object_ids()
        scene_name_a = self.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self.get_random_single_object_scene_name(object_id_b)
        
        metadata["object_id_a"]  = object_id_a
        metadata["scene_name_a"] = scene_name_a
        metadata["object_id_b"]  = object_id_b
        metadata["scene_name_b"] = scene_name_b
        metadata["type"] = SpartanDatasetDataType.DIFFERENT_OBJECT
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)

    def get_across_scene_data(self, scene_name_a, scene_name_b, metadata):
        """
        Essentially just returns a bunch of samples off the masks from scene_name_a, and scene_name_b.

        Since this data is across scene, we can't generate matches.

        Return args are for returning directly from __getitem__

        See get_within_scene_data() for documentation of return args.

        :param scene_name_a, scene_name_b: Names of scenes from which to each randomly sample an image
        :type scene_name_a, scene_name_b: strings
        :param metadata: a dict() holding metadata of the image pair, both for logging and for different downstream loss functions
        :type metadata: dict()
        """

        SD = SpartanDataset

        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")

        image_a_idx = self.get_random_image_index(scene_name_a)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name_a, image_a_idx)

        metadata['image_a_idx'] = image_a_idx

        # image b
        image_b_idx = self.get_random_image_index(scene_name_b)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name_b, image_b_idx)
        metadata['image_b_idx'] = image_b_idx

        # sample random indices from mask in image a
        num_samples = self.cross_scene_num_samples
        blind_uv_a = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_a_mask), num_samples)
        # sample random indices from mask in image b
        blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_b_mask), num_samples)

        if (blind_uv_a[0] is None) or (blind_uv_b[0] is None):
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)

        # data augmentation
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)

        if not self.debug:
            [image_a_rgb, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_mask], blind_uv_b)
        else:  # also mutate depth just for plotting
            [image_a_rgb, image_a_depth, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation(
                [image_a_rgb, image_a_depth, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_depth, image_b_mask], blind_uv_b)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]
        
        blind_uv_a_flat = SD.flatten_uv_tensor(blind_uv_a, image_width)
        blind_uv_b_flat = SD.flatten_uv_tensor(blind_uv_b, image_width)

        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        empty_tensor = SD.empty_tensor()

        if self.debug and ((blind_uv_a[0] is not None) and (blind_uv_b[0] is not None)):
            import correspondence_plotter
            num_matches_to_plot = 10

            plot_blind_uv_a, plot_blind_uv_b = SD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot*10)

            correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy, 
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_blind_uv_a, plot_blind_uv_b,
                                                                   circ_color='k', show=True)

        return metadata["type"], image_a_rgb, image_b_rgb, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, blind_uv_a_flat, blind_uv_b_flat, metadata

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
    def flatten_uv_tensor(uv_tensor, image_width):
        """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
        """
        return uv_tensor[1].long() * image_width + uv_tensor[0].long()

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]. It has a 1 exactly
        at the indices specified by uv_flat_tensor
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width*image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat


    @staticmethod
    def subsample_tuple(uv, num_samples):
        """
        Subsamples a tuple of (torch.Tensor, torch.Tensor)
        """
        indexes_to_keep = (torch.rand(num_samples) * len(uv[0])).floor().type(torch.LongTensor)
        return (torch.index_select(uv[0], 0, indexes_to_keep), torch.index_select(uv[1], 0, indexes_to_keep))

    @staticmethod
    def subsample_tuple_pair(uv_a, uv_b, num_samples):
        """
        Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)
        """
        assert len(uv_a[0]) == len(uv_b[0])
        indexes_to_keep = (torch.rand(num_samples) * len(uv_a[0])).floor().type(torch.LongTensor)
        uv_a_downsampled = (torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep))
        uv_b_downsampled = (torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep))
        return uv_a_downsampled, uv_b_downsampled


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
                                   'dataset', 'composite',
                                   'caterpillar_only.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset