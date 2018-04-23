#!/usr/bin/python


import os
import dense_correspondence_manipulation.utils.utils as utils
import logging
utils.add_dense_correspondence_to_python_path()
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import scipy.stats as ss

import torch
from torch.autograd import Variable
from torchvision import transforms


from dense_correspondence_manipulation.utils.constants import *
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork, NetworkMode
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

import dense_correspondence.evaluation.plotting as dc_plotting

from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image

class PandaDataFrameWrapper(object):
    """
    A simple wrapper for a PandaSeries that protects from read/write errors
    """

    def __init__(self, columns):
        data = [np.nan] * len(columns)
        self._columns = columns
        self._df = pd.DataFrame(data=[data], columns=columns)

    def set_value(self, key, value):
        if key not in self._columns:
            raise KeyError("%s is not in the index" %(key))

        self._df[key] = value

    def get_value(self, key):
        return self._df[key]

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, value):
        self._series = value

class DCNEvaluationPandaTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
             'img_a_idx',
             'img_b_idx',
            'is_valid',
            'norm_diff_descriptor_ground_truth',
            'norm_diff_descriptor',
            'norm_diff_ground_truth_3d',
            'norm_diff_pred_3d',
            'pixel_match_error_l2',
            'pixel_match_error_l1',
            'fraction_pixels_closer_than_ground_truth',
            'average_l2_distance_for_false_positives']

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNEvaluationPandaTemplate.columns)

class SIFTKeypointMatchPandaTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
               'img_a_idx',
               'img_b_idx',
               'is_valid',
               'norm_diff_pred_3d']

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, SIFTKeypointMatchPandaTemplate.columns)


class DenseCorrespondenceEvaluation(object):
    """
    Samples image pairs from the given scenes. Then uses the network to compute dense
    descriptors. Records the results of this in a Pandas.DataFrame object.
    """


    def __init__(self, config):
        self._config = config
        self._dataset = None


    def load_network_from_config(self, name):
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))


        network_config = self._config["networks"][name]
        return DenseCorrespondenceNetwork.from_config(network_config)

    def load_dataset_for_network(self, network_name):
        """
        Loads a dataset for the network specified in the config file
        :param network_name: string
        :type network_name:
        :return: SpartanDataset
        :rtype:
        """
        if network_name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(network_name))

        network_folder = os.path.dirname(self._config["networks"][network_name]["path_to_network_params"])
        network_folder = utils.convert_to_absolute_path(network_folder)
        dataset_config = utils.getDictFromYamlFilename(os.path.join(network_folder, "dataset.yaml"))

        dataset = SpartanDataset(config=dataset_config)
        return dataset


    def load_dataset(self):
        """
        Loads a SpartanDatasetMasked object
        For now we use a default one
        :return:
        :rtype: SpartanDatasetMasked
        """

        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset',
                                   'spartan_dataset_masked.yaml')

        config = utils.getDictFromYamlFilename(config_file)

        dataset = SpartanDataset(mode="test", config=config)

        return dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def get_output_dir(self):
        return utils.convert_to_absolute_path(self._config['output_dir'])

    @staticmethod
    def get_image_pair_with_poses_diff_above_threshold(dataset, scene_name, threshold=0.05,
                                                       max_num_attempts=100):
        """
        Given a dataset and scene name find a random pair of images with
        poses that are different above a threshold
        :param dataset:
        :type dataset:
        :param scene_name:
        :type scene_name:
        :param threshold:
        :type threshold:
        :param max_num_attempts:
        :type max_num_attempts:
        :return:
        :rtype:
        """
        img_a_idx = dataset.get_random_image_index(scene_name)
        pose_a = dataset.get_pose_from_scene_name_and_idx(scene_name, img_a_idx)
        pos_a = pose_a[0:3, 3]

        for i in xrange(0, max_num_attempts):
            img_b_idx = dataset.get_random_image_index(scene_name)
            pose_b = dataset.get_pose_from_scene_name_and_idx(scene_name, img_b_idx)
            pos_b = pose_b[0:3, 3]

            if np.linalg.norm(pos_a - pos_b) > threshold:
                return (img_a_idx, img_b_idx)

        return None


    def evaluate_single_network(self, network_name, mode="train", save=True):
        """
        Evaluates a single network, this network should be in the config
        :param network_name:
        :type network_name:
        :return:
        :rtype:
        """
        DCE = DenseCorrespondenceEvaluation

        dcn = self.load_network_from_config(network_name)
        dataset = self.dataset

        if mode == "train":
            dataset.set_train_mode()
        if mode == "test":
            dataset.set_test_mode()

        num_image_pairs = self._config['params']['num_image_pairs']
        num_matches_per_image_pair = self._config['params']['num_matches_per_image_pair']

        pd_dataframe_list, df = DCE.evaluate_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)


        # save pandas.DataFrame to csv
        if save:
            output_dir = os.path.join(self.get_output_dir(), network_name, mode)
            data_file = os.path.join(output_dir, "data.csv")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            df.to_csv(data_file)


    def evaluate_single_network_cross_scene(self, network_name, save=True):
        """
        This will search for the "evaluation_labeled_data_path" in the dataset.yaml,
        and use pairs of images that have been human-labeled across scenes.
        """

        dcn = self.load_network_from_config(network_name)
        dataset = dcn.load_training_dataset()

        if "evaluation_labeled_data_path" not in dataset.config:
            print "Could not find labeled cross scene data for this dataset."
            print "It needs to be set in the dataset.yaml of the folder from which"
            print "this network is loaded from."
            return

        cross_scene_data_path = dataset.config["evaluation_labeled_data_path"]
        home = os.path.dirname(utils.getDenseCorrespondenceSourceDir())
        cross_scene_data_full_path = os.path.join(home, cross_scene_data_path)
        cross_scene_data = utils.getDictFromYamlFilename(cross_scene_data_full_path)
        
        pd_dataframe_list = []
        for annotated_pair in cross_scene_data:

            scene_name_a = annotated_pair["image_a"]["scene_name"]
            scene_name_b = annotated_pair["image_b"]["scene_name"] 

            image_a_idx = annotated_pair["image_a"]["image_idx"]
            image_b_idx = annotated_pair["image_b"]["image_idx"]

            img_a_pixels = annotated_pair["image_a"]["pixels"]
            img_b_pixels = annotated_pair["image_b"]["pixels"]

            dataframe_list_temp =\
                DenseCorrespondenceEvaluation.single_cross_scene_image_pair_quantitative_analysis(dcn,
                dataset, scene_name_a, image_a_idx, scene_name_b, image_b_idx,
                img_a_pixels, img_b_pixels)

            assert dataframe_list_temp is not None

            import copy
            pd_dataframe_list += copy.copy(dataframe_list_temp)


        df = pd.concat(pd_dataframe_list)
        # save pandas.DataFrame to csv
        if save:
            output_dir = os.path.join(self.get_output_dir(), network_name, "cross-scene")
            data_file = os.path.join(output_dir, "data.csv")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            df.to_csv(data_file)


    @staticmethod
    def evaluate_network(dcn, dataset, num_image_pairs=25, num_matches_per_image_pair=100):
        """

        :param nn: A neural network DenseCorrespondenceNetwork
        :param test_dataset: DenseCorrespondenceDataset
            the dataset to draw samples from
        :return:
        """
        DCE = DenseCorrespondenceEvaluation

        logging_rate = 5


        pd_dataframe_list = []
        for i in xrange(0, num_image_pairs):


            scene_name = dataset.get_random_scene_name()

            # grab random scene
            if i % logging_rate == 0:
                print "computing statistics for image %d of %d, scene_name %s" %(i, num_image_pairs, scene_name)
                print "scene"


            idx_pair = DCE.get_image_pair_with_poses_diff_above_threshold(dataset, scene_name)

            if idx_pair is None:
                logging.info("no satisfactory image pair found, continuing")
                continue

            img_idx_a, img_idx_b = idx_pair

            dataframe_list_temp =\
                DCE.single_same_scene_image_pair_quantitative_analysis(dcn, dataset, scene_name,
                                                            img_idx_a,
                                                            img_idx_b,
                                                            num_matches=num_matches_per_image_pair,
                                                            debug=False)

            if dataframe_list_temp is None:
                print "no matches found, skipping"
                continue

            pd_dataframe_list += dataframe_list_temp
            # pd_series_list.append(series_list_temp)


        df = pd.concat(pd_dataframe_list)
        return pd_dataframe_list, df

    @staticmethod
    def plot_descriptor_colormaps(res_a, res_b, descriptor_image_stats=None,
                                  mask_a=None, mask_b=None, plot_masked=False):
        """
        Plots the colormaps of descriptors for a pair of images
        :param res_a: descriptors for img_a
        :type res_a: numpy.ndarray
        :param res_b:
        :type res_b: numpy.ndarray
        :return: None
        :rtype: None
        """

        if plot_masked:
            nrows = 2
            ncols = 2
        else:
            nrows = 1
            ncols = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(5)
        fig.set_figwidth(15)

        if descriptor_image_stats is None:
            res_a_norm, res_b_norm = dc_plotting.normalize_descriptor_pair(res_a, res_b)
        else:
            res_a_norm = dc_plotting.normalize_descriptor(res_a, descriptor_image_stats['entire_image'])
            res_b_norm = dc_plotting.normalize_descriptor(res_b, descriptor_image_stats['entire_image'])


        if plot_masked:
            ax = axes[0,0]
        else:
            ax = axes[0]

        ax.imshow(res_a_norm)


        if plot_masked:
            ax = axes[0,1]
        else:
            ax = axes[1]

        ax.imshow(res_b_norm)

        if plot_masked:
            assert mask_a is not None
            assert mask_b is not None

            fig.set_figheight(10)
            fig.set_figwidth(15)

            D = np.shape(res_a)[2]
            mask_a_repeat = np.repeat(mask_a[:,:,np.newaxis], D, axis=2)
            mask_b_repeat = np.repeat(mask_b[:,:,np.newaxis], D, axis=2)
            res_a_mask = mask_a_repeat * res_a
            res_b_mask = mask_b_repeat * res_b 

            if descriptor_image_stats is None:
                res_a_norm_mask, res_b_norm_mask = dc_plotting.normalize_descriptor_pair(res_a_mask, res_b_mask)
            else:
                res_a_norm_mask = dc_plotting.normalize_descriptor(res_a_mask, descriptor_image_stats['mask_image'])
                res_b_norm_mask = dc_plotting.normalize_descriptor(res_b_mask, descriptor_image_stats['mask_image'])

            axes[1,0].imshow(res_a_norm_mask)
            axes[1,1].imshow(res_b_norm_mask)

    @staticmethod
    def clip_pixel_to_image_size_and_round(uv, image_width, image_height):
        u = min(int(round(uv[0])), image_width - 1)
        v = min(int(round(uv[1])), image_height - 1)
        return (u,v)

    @staticmethod
    def single_cross_scene_image_pair_quantitative_analysis(dcn, dataset, scene_name_a,
                                               img_a_idx, scene_name_b, img_b_idx,
                                               img_a_pixels, img_b_pixels):
        """
        Quantitative analsys of a dcn on a pair of images from different scenes (requires human labeling).

        There is a bit of code copy from single_same_scene_image_pair_quantitative_analysis, but 
        it's a bit of a different structure, since matches are passed in and we need to try to generate more
        views of these sparse human labeled pixel matches.

        :param dcn: 
        :type dcn: DenseCorrespondenceNetwork
        :param dataset:
        :type dataset: SpartanDataset
        :param scene_name:
        :type scene_name: str
        :param img_a_idx:
        :type img_a_idx: int
        :param img_b_idx:
        :type img_b_idx: int
        :param img_a_pixels, img_b_pixels: lists of dicts, where each dict contains keys for "u" and "v"
                the lists should be the same length and index i from each list constitutes a match pair
        :return: Dict with relevant data
        :rtype:
        """

        rgb_a, depth_a, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)

        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

        depth_a = np.asarray(depth_a)
        depth_b = np.asarray(depth_b)
        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        camera_intrinsics_a = dataset.get_camera_intrinsics(scene_name_a)
        camera_intrinsics_b = dataset.get_camera_intrinsics(scene_name_b)
        if not np.allclose(camera_intrinsics_a.K, camera_intrinsics_b.K):
            print "Currently cannot handle two different camera K matrices in different scenes!"
            print "But you could add this..."
        camera_intrinsics_matrix = camera_intrinsics_a.K

        assert len(img_a_pixels) == len(img_b_pixels)

        print "Expanding amount of matches between:"
        print "scene_name_a", scene_name_a
        print "scene_name_b", scene_name_b
        print "originally had", len(img_a_pixels), "matches"
        
        image_height, image_width = dcn.image_shape
        DCE = DenseCorrespondenceEvaluation

        dataframe_list = []

        for i in range(len(img_a_pixels)):
            print "now, index of pixel match:", i
            uv_a = (img_a_pixels[i]["u"], img_a_pixels[i]["v"])
            uv_b = (img_b_pixels[i]["u"], img_b_pixels[i]["v"])
            uv_a = DCE.clip_pixel_to_image_size_and_round(uv_a, image_width, image_height)
            uv_b = DCE.clip_pixel_to_image_size_and_round(uv_b, image_width, image_height)
            print uv_a
            print uv_b

            # Reminder: this function wants only a single uv_a, uv_b
            pd_template = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(depth_a,
                                                                      depth_b,
                                                                      uv_a,
                                                                      uv_b,
                                                                      pose_a,
                                                                      pose_b,
                                                                      res_a,
                                                                      res_b,
                                                                      camera_intrinsics_matrix,
                                                                      rgb_a=rgb_a,
                                                                      rgb_b=rgb_b,
                                                                      debug=False)
            pd_template.set_value('scene_name', scene_name_a+"+"+scene_name_b)
            pd_template.set_value('img_a_idx', int(img_a_idx))
            pd_template.set_value('img_b_idx', int(img_b_idx))

            dataframe_list.append(pd_template.dataframe)

        return dataframe_list

    @staticmethod
    def single_same_scene_image_pair_quantitative_analysis(dcn, dataset, scene_name,
                                                img_a_idx, img_b_idx,
                                                camera_intrinsics_matrix=None,
                                                num_matches=100,
                                                debug=False):
        """
        Quantitative analysis of a dcn on a pair of images from the same scene.

        :param dcn: 
        :type dcn: DenseCorrespondenceNetwork
        :param dataset:
        :type dataset: SpartanDataset
        :param scene_name:
        :type scene_name: str
        :param img_a_idx:
        :type img_a_idx: int
        :param img_b_idx:
        :type img_b_idx: int
        :param camera_intrinsics_matrix: Optionally set camera intrinsics, otherwise will get it from the dataset
        :type camera_intrinsics_matrix: 3 x 3 numpy array
        :return: Dict with relevant data
        :rtype:
        """

        rgb_a, depth_a, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name, img_a_idx)

        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name, img_b_idx)

        depth_a = np.asarray(depth_a)
        depth_b = np.asarray(depth_b)
        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        if camera_intrinsics_matrix is None:
            camera_intrinsics = dataset.get_camera_intrinsics(scene_name)
            camera_intrinsics_matrix = camera_intrinsics.K

        # find correspondences
        (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(depth_a, pose_a, depth_b, pose_b,
                                                               device='CPU', img_a_mask=mask_a)

        if uv_a_vec is None:
            print "no matches found, returning"
            return None

        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []

        total_num_matches = len(uv_a_vec[0])
        match_list = random.sample(range(0, total_num_matches), num_matches)

        if debug:
            match_list = [50]

        logging_rate = 100

        image_height, image_width = dcn.image_shape

        DCE = DenseCorrespondenceEvaluation

        for i in match_list:
            uv_a = (uv_a_vec[0][i], uv_a_vec[1][i])
            uv_b_raw = (uv_b_vec[0][i], uv_b_vec[1][i])
            uv_b = DCE.clip_pixel_to_image_size_and_round(uv_b_raw, image_width, image_height)

            pd_template = DCE.compute_descriptor_match_statistics(depth_a,
                                                                  depth_b,
                                                                  uv_a,
                                                                  uv_b,
                                                                  pose_a,
                                                                  pose_b,
                                                                  res_a,
                                                                  res_b,
                                                                  camera_intrinsics_matrix,
                                                                  rgb_a=rgb_a,
                                                                  rgb_b=rgb_b,
                                                                  debug=debug)

            pd_template.set_value('scene_name', scene_name)
            pd_template.set_value('img_a_idx', int(img_a_idx))
            pd_template.set_value('img_b_idx', int(img_b_idx))

            dataframe_list.append(pd_template.dataframe)

        return dataframe_list

    @staticmethod
    def is_depth_valid(depth):
        """
        Checks if depth value is valid, usually missing depth values are either 0 or MAX_RANGE
        :param depth: depth in meters
        :type depth:
        :return:
        :rtype: bool
        """

        MAX_DEPTH = 10.0

        return ((depth > 0) and (depth < MAX_DEPTH))


    @staticmethod
    def compute_descriptor_match_statistics(depth_a, depth_b, uv_a, uv_b, pose_a, pose_b,
                                            res_a, res_b, camera_matrix, params=None,
                                            rgb_a=None, rgb_b=None, debug=False):
        """
        Computes statistics of descriptor pixelwise match.

        :param uv_a: a single pixel index in (u,v) coordinates, from image a
        :type uv_a: tuple of 2 ints
        :param uv_b: a single pixel index in (u,v) coordinates, from image b
        :type uv_b: tuple of 2 ints
        :param camera_matrix: camera intrinsics matrix
        :type camera_matrix: 3 x 3 numpy array
        :param rgb_a:
        :type rgb_a:
        :param rgb_b:
        :type rgb_b:
        :param depth_a: depth is assumed to be in mm (see conversion to meters below)
        :type depth_a: numpy array
        :param depth_b:
        :type depth_b:
        :param pose_a:
        :type pose_a: 4 x 4 numpy array
        :param pose_b:
        :type pose_b:
        :param res_a: descriptor for image a, of shape (H,W,D)
        :type res_a: numpy array
        :param res_b: descriptor for image b, of shape (H,W,D)
        :type res_b: numpy array
        :param params:
        :type params:
        :param debug: whether or not to print visualization
        :type debug:
        :return:
        :rtype:
        """

        DCE = DenseCorrespondenceEvaluation

        # compute best match
        uv_b_pred, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b, debug=debug)

        
        # compute pixel space difference
        pixel_match_error_l2 = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=2)
        pixel_match_error_l1 = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=1)


        # extract the ground truth descriptors
        des_a = res_a[uv_a[1], uv_a[0], :]
        des_b_ground_truth = res_b[uv_b[1], uv_b[0], :]
        norm_diff_descriptor_ground_truth = np.linalg.norm(des_a - des_b_ground_truth)

        # from Schmidt et al 2017: 
        """
        We then determine the number of pixels in the target image that are closer in
        descriptor space to the source point than the manually-labelled corresponding point.
        """
        # compute this
        (v_indices_better_than_ground_truth, u_indices_better_than_ground_truth) = np.where(norm_diffs < norm_diff_descriptor_ground_truth)
        num_pixels_closer_than_ground_truth = len(u_indices_better_than_ground_truth) 
        num_pixels_in_image = res_a.shape[0] * res_a.shape[1]
        fraction_pixels_closer_than_ground_truth = num_pixels_closer_than_ground_truth*1.0/num_pixels_in_image

        # new metric: average l2 distance of the pixels better than ground truth
        if num_pixels_closer_than_ground_truth == 0:
            average_l2_distance_for_false_positives = 0.0
        else:
            l2_distances = np.sqrt((u_indices_better_than_ground_truth - uv_b[0])**2 + (v_indices_better_than_ground_truth - uv_b[1])**2)
            average_l2_distance_for_false_positives = np.average(l2_distances)

        # extract depth values, note the indexing order of u,v has to be reversed
        uv_a_depth = depth_a[uv_a[1], uv_a[0]] / DEPTH_IM_SCALE # check if this is not None
        uv_b_depth = depth_b[uv_b[1], uv_b[0]] / DEPTH_IM_SCALE
        uv_b_pred_depth = depth_b[uv_b_pred[1], uv_b_pred[0]] / DEPTH_IM_SCALE
        uv_b_pred_depth_is_valid = DenseCorrespondenceEvaluation.is_depth_valid(uv_b_pred_depth)
        is_valid = uv_b_pred_depth_is_valid

        uv_a_pos = DCE.compute_3d_position(uv_a, uv_a_depth, camera_matrix, pose_a)
        uv_b_pos = DCE.compute_3d_position(uv_b, uv_b_depth, camera_matrix, pose_b)
        uv_b_pred_pos = DCE.compute_3d_position(uv_b_pred, uv_b_pred_depth, camera_matrix, pose_b)

        diff_ground_truth_3d = uv_b_pos - uv_a_pos

        diff_pred_3d = uv_b_pos - uv_b_pred_pos

        if DCE.is_depth_valid(uv_b_depth):
            norm_diff_ground_truth_3d = np.linalg.norm(diff_ground_truth_3d)
        else:
            norm_diff_ground_truth_3d = np.nan

        if DCE.is_depth_valid(uv_b_depth) and DCE.is_depth_valid(uv_b_pred_depth):
            norm_diff_pred_3d = np.linalg.norm(diff_pred_3d)
        else:
            norm_diff_pred_3d = np.nan

        if debug:

            print "uv_b_pred_depth is valid: "
            print "uv_a_pos: ", uv_a_pos
            print "uv_b_pos: ", uv_b_pos
            print "uv_b_pred_pos ", uv_b_pred_pos

            fig, axes = correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b,
                                                               uv_a, uv_b, show=False)

            correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b,
                                                               uv_a, uv_b_pred,
                                                               use_previous_plot=(fig, axes),
                                                               show=True,
                                                               circ_color='purple')

        pd_template = DCNEvaluationPandaTemplate()
        pd_template.set_value('norm_diff_descriptor', best_match_diff)
        pd_template.set_value('is_valid', is_valid)

        pd_template.set_value('norm_diff_ground_truth_3d', norm_diff_ground_truth_3d)

        if is_valid:
            pd_template.set_value('norm_diff_pred_3d', norm_diff_pred_3d)
        else:
            pd_template.set_value('norm_diff_pred_3d', np.nan)

        pd_template.set_value('norm_diff_descriptor_ground_truth', norm_diff_descriptor_ground_truth)

        pd_template.set_value('pixel_match_error_l2', pixel_match_error_l2)
        pd_template.set_value('pixel_match_error_l1', pixel_match_error_l1)

        pd_template.set_value('fraction_pixels_closer_than_ground_truth', fraction_pixels_closer_than_ground_truth)
        pd_template.set_value('average_l2_distance_for_false_positives', average_l2_distance_for_false_positives)


        return pd_template

    @staticmethod
    def compute_3d_position(uv, depth, camera_intrinsics_matrix, camera_to_world):
        """


        :param uv: pixel-location in (row, column) ordering
        :type uv:
        :param depth: depth-value
        :type depth:
        :param camera_intrinsics_matrix: the camera intrinsics matrix
        :type camera_intrinsics_matrix:
        :param camera_to_world: camera to world transform as a homogenous transform matrix
        :type camera_to_world: 4 x 4 numpy array
        :return:
        :rtype: np.array with shape (3,)
        """
        pos_in_camera_frame = correspondence_finder.pinhole_projection_image_to_world(uv, depth, camera_intrinsics_matrix)

        pos_in_world_frame = np.dot(camera_to_world, np.append(pos_in_camera_frame, 1))[:3]

        return pos_in_world_frame

    @staticmethod
    def single_same_scene_image_pair_qualitative_analysis(dcn, dataset, scene_name,
                                               img_a_idx, img_b_idx,
                                               num_matches=10):
        """
        Wrapper for single_image_pair_qualitative_analysis, when images are from same scene.

        See that function for remaining documentation.

        :param scene_name: scene name to use
        :param img_a_idx: index of image_a in the dataset
        :param img_b_idx: index of image_b in the datset

        :type scene_name: str
        :type img_a_idx: int
        :type img_b_idx: int

        :return: None
        """

        rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name, img_a_idx)

        rgb_b, _, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name, img_b_idx)

        DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn, dataset, rgb_a, rgb_b, mask_a, mask_b, num_matches)

    @staticmethod
    def single_cross_scene_image_pair_qualitative_analysis(dcn, dataset, scene_name_a,
                                               img_a_idx, scene_name_b, img_b_idx,
                                               num_matches=10):
        """
        Wrapper for single_image_pair_qualitative_analysis, when images are NOT from same scene.

        See that function for remaining documentation.

        :param scene_name: scene name to use
        :param img_a_idx: index of image_a in the dataset
        :param img_b_idx: index of image_b in the datset

        :type scene_name: str
        :type img_a_idx: int
        :type img_b_idx: int

        :return: the images a and b
        :rtype: PIL.Image, PIL.Image
        """

        rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)

        rgb_b, _, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

        DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn, dataset, rgb_a, rgb_b, mask_a, mask_b, num_matches)
        return rgb_a, rgb_b

    @staticmethod
    def single_image_pair_qualitative_analysis(dcn, dataset, rgb_a, rgb_b, mask_a, mask_b,
                                               num_matches):
        """
        Computes qualtitative assessment of DCN performance for a pair of
        images

        :param dcn: dense correspondence network to use
        :param dataset: dataset to use
        :param rgb_a, rgb_b: two rgb images for which to do matching
        :param mask_a, mask_b: masks of these two images
        :param num_matches: number of matches to generate
        
        :type dcn: DenseCorrespondenceNetwork
        :type dataset: DenseCorrespondenceDataset
        :type rgb_a, rgb_b: PIL.Images
        :type mask_a, mask_b: PIL.Images
        :type num_matches: int

        :return: None
        """

        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()


        # sample points on img_a. Compute best matches on img_b
        # note that this is in (x,y) format
        sampled_idx_list = random_sample_from_masked_image(mask_a, num_matches)

        # list of cv2.KeyPoint
        kp1 = []
        kp2 = []
        matches = []  # list of cv2.DMatch

        # placeholder constants for opencv
        diam = 0.01
        dist = 0.01

        try:
            descriptor_image_stats = dcn.descriptor_image_stats
        except:
            print "Could not find descriptor image stats..."
            print "Only normalizing pairs of images!" 
            descriptor_image_stats = None

        for i in xrange(0, num_matches):
            # convert to (u,v) format
            pixel_a = [sampled_idx_list[1][i], sampled_idx_list[0][i]]
            best_match_uv, best_match_diff, norm_diffs =\
                DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a,
                                                                                                     res_b)

            # be careful, OpenCV format is  (u,v) = (right, down)
            kp1.append(cv2.KeyPoint(pixel_a[0], pixel_a[1], diam))
            kp2.append(cv2.KeyPoint(best_match_uv[0], best_match_uv[1], diam))
            matches.append(cv2.DMatch(i, i, dist))

        gray_a_numpy = cv2.cvtColor(np.asarray(rgb_a), cv2.COLOR_BGR2GRAY)
        gray_b_numpy = cv2.cvtColor(np.asarray(rgb_b), cv2.COLOR_BGR2GRAY)
        img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, matches, flags=2, outImg=gray_b_numpy)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        axes.imshow(img3)

        # show colormap if possible (i.e. if descriptor dimension is 1 or 3)
        if dcn.descriptor_dimension in [1,3]:
            DenseCorrespondenceEvaluation.plot_descriptor_colormaps(res_a, res_b,
                                                                    descriptor_image_stats=descriptor_image_stats,
                                                                    mask_a=mask_a,
                                                                    mask_b=mask_b,
                                                                    plot_masked=True)

        plt.show()

    @staticmethod
    def compute_sift_keypoints(img, mask=None):
        """
        Compute SIFT keypoints given a grayscale img
        :param img:
        :type img:
        :param mask:
        :type mask:
        :return:
        :rtype:
        """

        # convert to grayscale image if needed
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, mask)
        kp, des = sift.compute(gray, kp)
        img_w_kp = 0 * img
        cv2.drawKeypoints(gray, kp, img_w_kp)
        return kp, des, gray, img_w_kp


    @staticmethod
    def single_image_pair_sift_analysis(dataset, scene_name,
                                        img_a_idx, img_b_idx,
                                        cross_match_threshold=0.75,
                                        num_matches=10,
                                        visualize=True,
                                        camera_intrinsics_matrix=None):
        """
        Computes SIFT features and does statistics
        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param scene_name:
        :type scene_name:
        :param img_a_idx:
        :type img_a_idx:
        :param img_b_idx:
        :type img_b_idx:
        :param num_matches:
        :type num_matches:
        :return:
        :rtype:
        """

        DCE = DenseCorrespondenceEvaluation

        rgb_a, depth_a, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name, img_a_idx)
        rgb_a = np.array(rgb_a) # converts PIL image to rgb

        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name, img_b_idx)
        rgb_b = np.array(rgb_b) # converts PIL image to rgb

        kp1, des1, gray1, img_1_kp = DCE.compute_sift_keypoints(rgb_a, mask_a)
        kp2, des2, gray2, img_2_kp = DCE.compute_sift_keypoints(rgb_b, mask_b)


        img1 = gray1
        img2 = gray2

        if visualize:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            axes[0].imshow(img_1_kp)
            axes[1].imshow(img_2_kp)
            plt.title("SIFT Keypoints")
            plt.show()

        # compute matches
        # Match descriptors.
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)  # Sort them in the order of their distance.
        total_num_matches = len(matches)

        # Apply ratio test
        good = []
        for m, n in matches:
            # m is the best match
            # n is the second best match

            if m.distance < 0.75 * n.distance:
                good.append([m])


        if visualize:
            good_vis = random.sample(good, 5)
            outImg = 0 * img1 # placeholder
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_vis, outImg, flags=2)
            plt.imshow(img3)
            plt.title("SIFT Keypoint Matches")
            plt.show()


        if camera_intrinsics_matrix is None:
            camera_intrinsics = dataset.get_camera_intrinsics(scene_name)
            camera_intrinsics_matrix = camera_intrinsics.K

        dataframe_list = []

        for idx, val in enumerate(good):
            match = val[0]
            kp_a = kp1[match.queryIdx]
            kp_b = kp2[match.trainIdx]
            df = DCE.compute_single_sift_match_statistics(depth_a, depth_b, kp_a, kp_b,
                                                     pose_a, pose_b, camera_intrinsics_matrix)

            dataframe_list.append(df)



        returnData = dict()
        returnData['kp1'] = kp1
        returnData['kp2'] = kp2
        returnData['matches'] = matches
        returnData['good'] = good
        returnData['dataframe_list'] = dataframe_list

        return returnData





    @staticmethod
    def compute_single_sift_match_statistics(depth_a, depth_b, kp_a, kp_b, pose_a, pose_b,
                                            camera_matrix, params=None,
                                            rgb_a=None, rgb_b=None, debug=False):
        """
        Compute some statistics of the SIFT match

        :param depth_a:
        :type depth_a:
        :param depth_b:
        :type depth_b:
        :param kp_a: kp_a.pt is the (u,v) = (column, row) coordinates in the image
        :type kp_a: cv2.KeyPoint
        :param kp_b:
        :type kp_b:
        :param pose_a:
        :type pose_a:
        :param pose_b:
        :type pose_b:
        :param camera_matrix:
        :type camera_matrix:
        :param params:
        :type params:
        :param rgb_a:
        :type rgb_a:
        :param rgb_b:
        :type rgb_b:
        :param debug:
        :type debug:
        :return:
        :rtype:
        """

        DCE = DenseCorrespondenceEvaluation
        # first compute location of kp_a in world frame

        image_height, image_width = depth_a.shape[0], depth_a.shape[1]

        def clip_pixel_to_image_size_and_round(uv):
            u = min(int(round(uv[0])), image_width - 1)
            v = min(int(round(uv[1])), image_height - 1)
            return [u,v]

        uv_a = clip_pixel_to_image_size_and_round((kp_a.pt[0], kp_a.pt[1]))
        uv_a_depth = depth_a[uv_a[1], uv_a[0]] / DEPTH_IM_SCALE
        # print "uv_a", uv_a
        # print "uv_a_depth", uv_a_depth
        # print "camera_matrix", camera_matrix
        # print "pose_a", pose_a
        kp_a_3d = DCE.compute_3d_position(uv_a, uv_a_depth, camera_matrix, pose_a)


        uv_b = clip_pixel_to_image_size_and_round((kp_b.pt[0], kp_b.pt[1]))
        uv_b_depth = depth_b[uv_b[1], uv_b[0]] / DEPTH_IM_SCALE
        uv_b_depth_valid = DCE.is_depth_valid(uv_b_depth)
        kp_b_3d = DCE.compute_3d_position(uv_b, uv_b_depth, camera_matrix, pose_b)


        # uv_b_ground_truth = correspondence_finder.pinhole_projection_world_to_image(kp_b_3d, camera_matrix, camera_to_world=pose_b)

        is_valid = uv_b_depth_valid

        if debug:
            print "\n\n"
            print "uv_a", uv_a
            print "kp_a_3d", kp_a_3d
            print "kp_b_3d", kp_b_3d
            print "is_valid", is_valid



        norm_diff_pred_3d = np.linalg.norm(kp_b_3d - kp_a_3d)

        pd_template = SIFTKeypointMatchPandaTemplate()
        pd_template.set_value('is_valid', is_valid)

        if is_valid:
            pd_template.set_value('norm_diff_pred_3d', norm_diff_pred_3d)

        return pd_template


    @staticmethod
    def single_image_pair_keypoint_analysis(dcn, dataset, scene_name,
                                                img_a_idx, img_b_idx,
                                                params=None,
                                                camera_intrinsics_matrix=None, visualize=True):

        DCE = DenseCorrespondenceEvaluation
        # first compute SIFT stuff
        sift_data = DCE.single_image_pair_sift_analysis(dataset, scene_name,
                                        img_a_idx, img_b_idx, visualize=visualize)

        kp1 = sift_data['kp1']
        kp2 = sift_data['kp2']

        rgb_a, depth_a, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name, img_a_idx)
        rgb_a = np.array(rgb_a)  # converts PIL image to rgb

        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name, img_b_idx)
        rgb_b = np.array(rgb_b)  # converts PIL image to rgb


        # compute the best matches among the SIFT keypoints
        des1 = dcn.evaluate_descriptor_at_keypoints(rgb_a, kp1)
        des2 = dcn.evaluate_descriptor_at_keypoints(rgb_b, kp2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)  # Sort them in the order of their distance.
        total_num_matches = len(matches)

        good = []
        for idx, val in enumerate(matches):
            m, n = val
            if (m.distance < 0.5 * n.distance) and m.distance < 0.01:
                print "\n\n"
                print "m.distance", m.distance
                print "n.distance", n.distance
                good.append([m])


            #
            # if idx > 5:
            #     return


        print "total keypoints = ", len(kp1)
        print "num good matches = ", len(good)
        print "SIFT good matches = ", len(sift_data['good'])
        if visualize:
            img1 = cv2.cvtColor(rgb_a, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(rgb_b, cv2.COLOR_BGR2GRAY)


            good_vis = random.sample(good, 5)
            outImg = 0 * img1 # placeholder
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_vis, outImg, flags=2)
            plt.imshow(img3)
            plt.title("Dense Correspondence Keypoint Matches")
            plt.show()

        returnData = dict()
        returnData['kp1'] = kp1
        returnData['kp2'] = kp2
        returnData['matches'] = matches
        returnData['des1'] = des1
        returnData['des2'] = des2

        return returnData
        return returnData


    @staticmethod
    def evaluate_network_qualitative_cross_scene(dcn, dataset, draw_human_annotations=True):
        """
        This will search for the "evaluation_labeled_data_path" in the dataset.yaml,
        and use pairs of images that have been human-labeled across scenes.
        """

        if "evaluation_labeled_data_path" not in dataset.config:
            print "Could not find labeled cross scene data for this dataset."
            print "It needs to be set in the dataset.yaml of the folder from which"
            print "this network is loaded from."
            return

        cross_scene_data_path = dataset.config["evaluation_labeled_data_path"]
        home = os.path.dirname(utils.getDenseCorrespondenceSourceDir())
        cross_scene_data_full_path = os.path.join(home, cross_scene_data_path)
        cross_scene_data = utils.getDictFromYamlFilename(cross_scene_data_full_path)
        
        for annotated_pair in cross_scene_data:

            scene_name_a = annotated_pair["image_a"]["scene_name"]
            scene_name_b = annotated_pair["image_b"]["scene_name"] 

            image_a_idx = annotated_pair["image_a"]["image_idx"]
            image_b_idx = annotated_pair["image_b"]["image_idx"]

            rgb_a, rgb_b = DenseCorrespondenceEvaluation.single_cross_scene_image_pair_qualitative_analysis(\
                dcn, dataset, scene_name_a, image_a_idx, scene_name_b, image_b_idx)


            if draw_human_annotations:
                img_a_points_picked = annotated_pair["image_a"]["pixels"]
                img_b_points_picked = annotated_pair["image_b"]["pixels"]

                # note here: converting the rgb_a to numpy format, but not inverting
                # the RGB <--> BGR colors as cv2 would expect, because all I'm going to do is then
                # plot this as an image in matplotlib, in which case
                # would just need to switch the colors back.
                rgb_a = dc_plotting.draw_correspondence_points_cv2(np.asarray(rgb_a), img_a_points_picked)
                rgb_b = dc_plotting.draw_correspondence_points_cv2(np.asarray(rgb_b), img_b_points_picked)

                fig, axes = plt.subplots(nrows=1, ncols=2)
                fig.set_figheight(10)
                fig.set_figwidth(15)
                axes[0].imshow(rgb_a)
                axes[1].imshow(rgb_b)
                plt.show()


    @staticmethod
    def evaluate_network_qualitative(dcn, dataset, num_image_pairs=5, randomize=False,
                                     scene_type="caterpillar"):


        # Train Data
        print "\n\n-----------Train Data Evaluation----------------"
        if randomize:
            raise NotImplementedError("not yet implemented")
        else:
            if scene_type == "caterpillar":
                scene_name = '2018-04-10-16-06-26'
                img_pairs = []
                img_pairs.append([0,753])
                img_pairs.append([812, 1218])
                img_pairs.append([1430, 1091])
                img_pairs.append([1070, 649])
            elif scene_type == "drill":
                scene_name = '13_drill_long_downsampled'
                img_pairs = []
                img_pairs.append([0, 737])
                img_pairs.append([409, 1585])
                img_pairs.append([2139, 1041])
                img_pairs.append([235, 1704])
            else:
                raise ValueError("scene_type must be one of [drill, caterpillar], it was %s)" %(scene_type))

        for img_pair in img_pairs:
            print "Image pair (%d, %d)" %(img_pair[0], img_pair[1])
            DenseCorrespondenceEvaluation.single_same_scene_image_pair_qualitative_analysis(dcn,
                                                                                 dataset,
                                                                                 scene_name,
                                                                                 img_pair[0],
                                                                                 img_pair[1])

        # Test Data
        print "\n\n-----------Test Data Evaluation----------------"
        dataset.set_test_mode()
        if randomize:
            raise NotImplementedError("not yet implemented")
        else:
            if scene_type == "caterpillar":
                scene_name = '2018-04-10-16-08-46'
                img_pairs = []
                img_pairs.append([0, 754])
                img_pairs.append([813, 1219])
                img_pairs.append([1429, 1092])
                img_pairs.append([1071, 637])
            elif scene_type == "drill":
                scene_name = '06_drill_long_downsampled'
                img_pairs = []
                img_pairs.append([0, 617])
                img_pairs.append([270, 786])
                img_pairs.append([1001, 2489])
                img_pairs.append([1536, 1917])
            else:
                raise ValueError("scene_type must be one of [drill, caterpillar], it was %s)" % (scene_type))


        for img_pair in img_pairs:
            print "Image pair (%d, %d)" %(img_pair[0], img_pair[1])
            DenseCorrespondenceEvaluation.single_same_scene_image_pair_qualitative_analysis(dcn,
                                                                                 dataset,
                                                                                 scene_name,
                                                                                 img_pair[0],
                                                                                 img_pair[1])

        if scene_type == "caterpillar":
            # Train Data
            print "\n\n-----------More Test Data Evaluation----------------"
            if randomize:
                raise NotImplementedError("not yet implemented")
            else:

                scene_name = '2018-04-16-14-25-19'
                img_pairs = []
                img_pairs.append([0,1553])
                img_pairs.append([1729, 2386])
                img_pairs.append([2903, 1751])
                img_pairs.append([841, 771])

            for img_pair in img_pairs:
                print "Image pair (%d, %d)" %(img_pair[0], img_pair[1])
                DenseCorrespondenceEvaluation.single_same_scene_image_pair_qualitative_analysis(dcn,
                                                                                 dataset,
                                                                                 scene_name,
                                                                                 img_pair[0],
                                                                                 img_pair[1])

    @staticmethod
    def compute_loss_on_dataset(dcn, data_loader, num_iterations=500):
        """

        Computes the loss for the given number of iterations

        :param dcn:
        :type dcn:
        :param data_loader:
        :type data_loader:
        :param num_iterations:
        :type num_iterations:
        :return:
        :rtype:
        """


        # loss_vec = np.zeros(num_iterations)
        loss_vec = []
        match_loss_vec = []
        non_match_loss_vec = []
        counter = 0
        pixelwise_contrastive_loss = PixelwiseContrastiveLoss()

        batch_size = 1

        for i, data in enumerate(data_loader, 0):

            # get the inputs
            data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b, metadata = data
            data_type = data_type[0]

            if len(matches_a[0]) == 0:
                print "didn't have any matches, continuing"
                continue

            img_a = Variable(img_a.cuda(), requires_grad=False)
            img_b = Variable(img_b.cuda(), requires_grad=False)

            if data_type == "matches":
                matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
                non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)

            # run both images through the network
            image_a_pred = dcn.forward(img_a)
            image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

            image_b_pred = dcn.forward(img_b)
            image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

            # get loss
            if data_type == "matches":
                loss, match_loss, non_match_loss = \
                    pixelwise_contrastive_loss.get_loss(image_a_pred,
                                                        image_b_pred,
                                                        matches_a,
                                                        matches_b,
                                                        non_matches_a,
                                                        non_matches_b)



                loss_vec.append(loss.data[0])
                non_match_loss_vec.append(non_match_loss.data[0])
                match_loss_vec.append(match_loss.data[0])


            if i > num_iterations:
                break

        loss_vec = np.array(loss_vec)
        match_loss_vec = np.array(match_loss_vec)
        non_match_loss_vec = np.array(non_match_loss_vec)

        loss = np.average(loss_vec)
        match_loss = np.average(match_loss_vec)
        non_match_loss = np.average(non_match_loss_vec)

        return loss, match_loss, non_match_loss



    @staticmethod
    def compute_descriptor_statistics_on_dataset(dcn, dataset, num_images=100,
                                                 save_to_file=True, filename=None):
        """
        Computes the statistics of the descriptors on the dataset
        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param save_to_file:
        :type save_to_file:
        :return:
        :rtype:
        """

        to_tensor = transforms.ToTensor()

        # compute the per-channel mean
        def compute_descriptor_statistics(res, mask_tensor):
            """
            Computes
            :param res: The output of the DCN
            :type res: torch.FloatTensor with shape [H,W,D]
            :return: min, max, mean
            :rtype: each is torch.FloatTensor of shape [D]
            """
            # convert to [W*H, D]
            D = res.shape[2]

            # convert to torch.FloatTensor instead of variable
            if isinstance(res, torch.autograd.Variable):
                res = res.data

            res_reshape = res.contiguous().view(-1,D)
            channel_mean = res_reshape.mean(0) # shape [D]
            channel_min, _ = res_reshape.min(0) # shape [D]
            channel_max, _ = res_reshape.max(0) # shape [D]

            # now do the same for the masked image
            mask_flat = mask_tensor.view(-1,1).squeeze(1)

            mask_indices_flat = torch.nonzero(mask_flat).squeeze(1)

            res_masked_flat = res_reshape.index_select(0, mask_indices_flat) # shape [mask_size, D]
            mask_channel_mean = res_masked_flat.mean(0)
            mask_channel_min, _ = res_masked_flat.min(0)
            mask_channel_max, _ = res_masked_flat.max(0)


            entire_image_stats = (channel_min, channel_max, channel_mean)
            mask_image_stats = (mask_channel_min, mask_channel_max, mask_channel_mean)
            return entire_image_stats, mask_image_stats

        def compute_descriptor_std_dev(res, channel_mean):
            """
            Computes the std deviation of a descriptor image, given a channel mean
            :param res:
            :type res:
            :param channel_mean:
            :type channel_mean:
            :return:
            :rtype:
            """
            D = res.shape[2]
            res_reshape = res.view(-1, D) # shape [W*H,D]
            v = res - channel_mean
            std_dev = torch.std(v, 0) # shape [D]
            return std_dev

        def update_stats(stats_dict, single_img_stats):
            """
            Update the running mean, min and max
            :param stats_dict:
            :type stats_dict:
            :param single_img_stats:
            :type single_img_stats:
            :return:
            :rtype:
            """

            min_temp, max_temp, mean_temp = single_img_stats

            if stats_dict['min'] is None:
                stats_dict['min'] = min_temp
            else:
                stats_dict['min'] = torch.min(stats_dict['min'], min_temp)

            if stats_dict['max'] is None:
                stats_dict['max'] = max_temp
            else:
                stats_dict['max'] = torch.max(stats_dict['max'], max_temp)

            if stats_dict['mean'] is None:
                stats_dict['mean'] = mean_temp
            else:
                stats_dict['mean'] += mean_temp


        stats = dict()
        stats['entire_image'] = {'mean': None, 'max': None, 'min': None}
        stats['mask_image'] = {'mean': None, 'max': None, 'min': None}

        for i in xrange(0,num_images):
            rgb, depth, mask, _ = dataset.get_random_rgbd_mask_pose()
            img_tensor = dataset.rgb_image_to_tensor(rgb)
            res = dcn.forward_single_image_tensor(img_tensor)  # [H, W, D]

            mask_tensor = to_tensor(mask).cuda()
            entire_image_stats, mask_image_stats = compute_descriptor_statistics(res, mask_tensor)

            update_stats(stats['entire_image'], entire_image_stats)
            update_stats(stats['mask_image'], mask_image_stats)


        for key, val in stats.iteritems():
            val['mean'] = 1.0/num_images * val['mean']
            for field in val:
                val[field] = val[field].tolist()

        if save_to_file:
            if filename is None:
                path_to_params_folder = dcn.config['path_to_network_params_folder']
                path_to_params_folder = utils.convert_to_absolute_path(path_to_params_folder)
                filename = os.path.join(path_to_params_folder, 'descriptor_statistics.yaml')

            utils.saveToYaml(stats, filename)



        return stats




    @staticmethod
    def make_default():
        """
        Makes a DenseCorrespondenceEvaluation object using the default config
        :return:
        :rtype: DenseCorrespondenceEvaluation
        """
        config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'evaluation.yaml')
        config = utils.getDictFromYamlFilename(config_filename)
        return DenseCorrespondenceEvaluation(config)

    ############ TESTING ################


    @staticmethod
    def test(dcn, dataset, data_idx=1, visualize=False, debug=False, match_idx=10):

        scene_name = '13_drill_long_downsampled'
        img_idx_a = utils.getPaddedString(0)
        img_idx_b = utils.getPaddedString(737)

        DenseCorrespondenceEvaluation.single_same_scene_image_pair_qualitative_analysis(dcn, dataset,
                                                                             scene_name, img_idx_a,
                                                                             img_idx_b)

class DenseCorrespondenceEvaluationPlotter(object):

    def __init__(self, config=None):
        if config is None:
            config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config',
                                       'dense_correspondence', 'evaluation',
                                       'evaluation_plotter.yaml')

            config = utils.getDictFromYamlFilename(config_file)

        self._config = config

    def load_dataframe(self, network_name):
        """
        Loads the specified dataframe for the given network specified in the config file
        :param network_name:
        :type network_name:
        :return:
        :rtype:
        """

        if network_name not in self._config['networks']:
            raise ValueError("%s not in config" %(network_name))

        path_to_csv = self._config['networks'][network_name]['path_to_csv']
        path_to_csv = utils.convert_to_absolute_path(path_to_csv)
        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)
        return df

    @staticmethod
    def make_cdf_plot(ax, data, num_bins, label=None):
        """
        Plots the empirical CDF of the data
        :param ax: axis of a matplotlib plot to plot on
        :param data:
        :type data:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        cumhist, l, b, e = ss.cumfreq(data, num_bins)
        cumhist *= 1.0 / len(data)
        x_axis = l + b * np.arange(0, num_bins)
        plot = ax.plot(x_axis, cumhist, label=label)
        return plot

    @staticmethod
    def make_pixel_match_error_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df: pandas dataframe, i.e. generated from quantitative 
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['pixel_match_error_l2']

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Pixel match error, L2 (pixel distance)')
        ax.set_ylabel('Fraction of images')
        return plot

    @staticmethod
    def make_descriptor_accuracy_plot(ax, df, label=None, num_bins=100):
        """
        Makes a plot of best match accuracy.
        Drops nans
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['norm_diff_pred_3d']
        data = data.dropna()
        data *= 100 # convert to cm

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('3D match error, L2 (cm)')
        ax.set_ylabel('Fraction of images')
        #ax.set_title("3D Norm Diff Best Match")
        return plot

    @staticmethod
    def make_norm_diff_ground_truth_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['norm_diff_descriptor_ground_truth']
        
        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Descriptor match error, L2')
        ax.set_ylabel('Fraction of images')
        return plot

    @staticmethod
    def make_fraction_false_positives_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['fraction_pixels_closer_than_ground_truth']
        
        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Fraction false positives')
        ax.set_ylabel('Fraction of images')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return plot

    @staticmethod
    def make_average_l2_false_positives_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['average_l2_distance_for_false_positives']
        
        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Average l2 pixel distance for false positives')
        ax.set_ylabel('Fraction of images')
        return plot

    @staticmethod
    def compute_area_above_curve(df, field, num_bins=100):
        """
        Computes AOC for the entries in that field
        :param df:
        :type df: Pandas.DataFrame
        :param field: specifies which column of the DataFrame to use
        :type field: str
        :return:
        :rtype:
        """

        data = df[field]
        data = data.dropna()

        cumhist, l, b, e = ss.cumfreq(data, num_bins)
        cumhist *= 1.0 / len(data)

        # b is bin width
        area_above_curve = b * np.sum((1-cumhist))
        return area_above_curve

    @staticmethod
    def run_on_single_dataframe(path_to_df_csv, label=None, output_dir=None, save=True, previous_fig_axes=None):
        """
        This method is intended to be called from an ipython notebook for plotting.

        Usage notes:
        - after calling this function, you can still change many things about the plot
        - for example you can still call plt.title("New title") to change the title
        - if you'd like to plot multiple lines on the same axes, then take the return arg of a previous call to this function, 
        - and pass it into previous_plot, i.e.:
            fig = run_on_single_dataframe("thing1.csv")
            run_on_single_dataframe("thing2.csv", previous_plot=fig)
            plt.title("both things")
            plt.show()
        - if you'd like each line to have a label in the plot, then use pass a string to label, i.e.:
            fig = run_on_single_dataframe("thing1.csv", label="thing1")
            run_on_single_dataframe("thing2.csv", label="thing2", previous_plot=fig)
            plt.title("both things")
            plt.show()

        :param path_to_df_csv: full path to csv file
        :type path_to_df_csv: string
        :param label: name that will show up labeling this line in the legend
        :type label: string
        :param save: whether or not you want to save a .png
        :type save: bool
        :param previous_plot: a previous matplotlib figure to keep building on
        :type previous_plot: None or matplotlib figure 
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        path_to_csv = utils.convert_to_absolute_path(path_to_df_csv)

        if output_dir is None:
            output_dir = os.path.dirname(path_to_csv)

        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)

        if previous_fig_axes==None:
            N = 5
            fig, axes = plt.subplots(N, figsize=(10,N*5))
        else:
            [fig, axes] = previous_fig_axes
        
        
        # pixel match error
        plot = DCEP.make_pixel_match_error_plot(axes[0], df, label=label)
        axes[0].legend()
       
        # 3D match error
        plot = DCEP.make_descriptor_accuracy_plot(axes[1], df, label=label)
        if save:
            fig_file = os.path.join(output_dir, "norm_diff_pred_3d.png")
            fig.savefig(fig_file)

        aac = DCEP.compute_area_above_curve(df, 'norm_diff_pred_3d')
        d = dict()
        d['norm_diff_3d_area_above_curve'] = float(aac)

        # norm difference of the ground truth match (should be 0)
        plot = DCEP.make_norm_diff_ground_truth_plot(axes[2], df, label=label)

        # fraction false positives
        plot = DCEP.make_fraction_false_positives_plot(axes[3], df, label=label)

        # average l2 false positives
        plot = DCEP.make_average_l2_false_positives_plot(axes[4], df, label=label)

        yaml_file = os.path.join(output_dir, 'stats.yaml')
        utils.saveToYaml(d, yaml_file)
        return [fig, axes]


def run():
    pass

def main(config):
    eval = DenseCorrespondenceEvaluation(config)
    dcn = eval.load_network_from_config("10_scenes_drill")
    test_dataset = SpartanDataset(mode="test")

    DenseCorrespondenceEvaluation.test(dcn, test_dataset)

def test():
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'evaluation', 'evaluation.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    default_config = utils.get_defaults_config()
    utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

    main(config)

if __name__ == "__main__":
    test()