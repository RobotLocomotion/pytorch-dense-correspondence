#!/usr/bin/python

## NOTE TO PETE:
# - to switch this between dynamic and static,
# - just ctrl+f for dynamic


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
import itertools
import yaml

import torch
from torch.autograd import Variable
from torchvision import transforms


from dense_correspondence_manipulation.utils.constants import *
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.dataset.dynamic_spartan_dataset import DynamicSpartanDataset

import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.network.dense_detection_network import DenseDetectionResnet
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import dense_correspondence_manipulation.utils.visualization as vis_utils

import dense_correspondence.evaluation.plotting as dc_plotting

from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image

from dense_correspondence.evaluation.utils import PandaDataFrameWrapper

# why don't we have scene_name
class DCNEvaluationPandaTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
               'scene_name_a',
               'scene_name_b',
               'object_id_a',
               'object_id_b',
               'img_a_idx',
               'img_b_idx',
               'is_valid',
               'is_valid_masked',
               'norm_diff_descriptor_ground_truth',
               'norm_diff_descriptor',
               'norm_diff_descriptor_masked',
               'norm_diff_ground_truth_3d',
               'norm_diff_pred_3d',
               'norm_diff_pred_3d_masked',
               'pixel_match_error_l2',
               'pixel_match_error_l2_masked',
               'pixel_match_error_l1',
               'fraction_pixels_closer_than_ground_truth',
               'fraction_pixels_closer_than_ground_truth_masked',
               'average_l2_distance_for_false_positives',
               'average_l2_distance_for_false_positives_masked',
               'keypoint_name' # (optional) name of the keypoint
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNEvaluationPandaTemplate.columns)

class DCNDetectionEvaluationPandaTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
               'img_a_idx',
               'img_b_idx',
               'best_match_diff',
               'true_positive'
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNDetectionEvaluationPandaTemplate.columns)


class DCNEvaluationPandaTemplateAcrossObject(PandaDataFrameWrapper):
    columns = ['scene_name_a',
            'scene_name_b',
            'img_a_idx',
            'img_b_idx',
            'object_id_a',
            'object_id_b',
            'norm_diff_descriptor_best_match']

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNEvaluationPandaTemplateAcrossObject.columns)

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


    @property
    def config(self):
        return self._configs

    def load_network_from_config(self, name):
        """
        Loads a network from config file. Puts it in eval mode by default
        :param name:
        :type name:
        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))


        path_to_network_params = self._config["networks"][name]["path_to_network_params"]
        path_to_network_params = utils.convert_data_relative_path_to_absolute_path(path_to_network_params, assert_path_exists=True)
        model_folder = os.path.dirname(path_to_network_params)

        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        dcn.eval()
        return dcn

    def load_detection_network_from_config(self, name):
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))
        path_to_network_params = self._config["networks"][name]["path_to_network_params"]
        path_to_network_params = utils.convert_data_relative_path_to_absolute_path(path_to_network_params, assert_path_exists=True)
        model_folder = os.path.dirname(path_to_network_params)
        detection_net = DenseDetectionResnet.from_model_folder(model_folder, model_param_file=path_to_network_params)
        detection_net.eval()
        return detection_net



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
        network_folder = utils.convert_data_relative_path_to_absolute_path(network_folder, assert_path_exists=True)
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
        return utils.convert_data_relative_path_to_absolute_path(self._config['output_dir'])

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
        dcn.eval()
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
        Simple wrapper that uses class config and then calls static method
        """
        dcn = self.load_network_from_config(network_name)
        dcn.eval()
        dataset = dcn.load_training_dataset()
        DenseCorrespondenceEvaluation.evaluate_network_cross_scene(dcn, dataset, save=save)

    @staticmethod
    def evaluate_network_cross_scene(dcn=None, dataset=None, save=True):
        """
        This will search for the "evaluation_labeled_data_path" in the dataset.yaml,
        and use pairs of images that have been human-labeled across scenes.
        """

        utils.reset_random_seed()

        cross_scene_data = DenseCorrespondenceEvaluation.parse_cross_scene_data(dataset)

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

            pd_dataframe_list += dataframe_list_temp


        df = pd.concat(pd_dataframe_list)
        # save pandas.DataFrame to csv
        if save:
            output_dir = os.path.join(self.get_output_dir(), network_name, "cross-scene")
            data_file = os.path.join(output_dir, "data.csv")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            df.to_csv(data_file)
        return df


    @staticmethod
    def evaluate_network_across_objects(dcn=None, dataset=None, num_image_pairs=25):
        """
        This grabs different objects and computes a small set of statistics on their distribution.
        """

        utils.reset_random_seed()

        pd_dataframe_list = []
        for i in xrange(num_image_pairs):

            object_id_a, object_id_b = dataset.get_two_different_object_ids()
            scene_name_a = dataset.get_random_single_object_scene_name(object_id_a)
            scene_name_b = dataset.get_random_single_object_scene_name(object_id_b)

            image_a_idx = dataset.get_random_image_index(scene_name_a)
            image_b_idx = dataset.get_random_image_index(scene_name_b)

            dataframe_list_temp =\
                DenseCorrespondenceEvaluation.single_across_object_image_pair_quantitative_analysis(dcn,
                dataset, scene_name_a, scene_name_b, image_a_idx, image_b_idx, object_id_a, object_id_b)

            # if the list is empty, don't bother +=ing it, just continue
            if len(dataframe_list_temp) == 0:
                continue

            assert dataframe_list_temp is not None

            pd_dataframe_list += dataframe_list_temp


        df = pd.concat(pd_dataframe_list)

        return df


    def evaluate_single_network_cross_instance(self, network_name, full_path_cross_instance_labels, save=False):
        """
        Simple wrapper that uses class config and then calls static method
        """
        dcn = self.load_network_from_config(network_name)
        dcn.eval()
        dataset = dcn.load_training_dataset()
        return DenseCorrespondenceEvaluation.evaluate_network_cross_instance(dcn, dataset, full_path_cross_instance_labels, save=False)

    @staticmethod
    def evaluate_network_cross_instance(dcn=None, dataset=None, full_path_cross_instance_labels=None, save=False):
        """
        This will grab the .yaml specified via its full path (full_path_cross_instance_labels)
        and use globally class-consistent keypoints that have been human-labeled across instances.
        """

        utils.reset_random_seed()

        cross_instance_keypoint_labels = utils.getDictFromYamlFilename(full_path_cross_instance_labels)

        print cross_instance_keypoint_labels

        # keypoints = dict()
        # for label in cross_instance_keypoint_labels:
        #     for keypoint_label in label['image']['pixels']:
        #         if keypoint_label['keypoint'] not in keypoints:
        #             print "Found new keypoint:", keypoint_label['keypoint']


        pd_dataframe_list = []

        # generate all pairs of images
        import itertools
        for subset in itertools.combinations(cross_instance_keypoint_labels, 2):
            print(subset)

            scene_name_a = subset[0]["image"]["scene_name"]
            scene_name_b = subset[1]["image"]["scene_name"]

            image_a_idx = subset[0]["image"]["image_idx"]
            image_b_idx = subset[1]["image"]["image_idx"]

            img_a_pixels = subset[0]["image"]["pixels"]
            img_b_pixels = subset[1]["image"]["pixels"]

            dataframe_list_temp =\
                DenseCorrespondenceEvaluation.single_cross_scene_image_pair_quantitative_analysis(dcn,
                dataset, scene_name_a, image_a_idx, scene_name_b, image_b_idx,
                img_a_pixels, img_b_pixels)

            assert dataframe_list_temp is not None

            pd_dataframe_list += dataframe_list_temp


        df = pd.concat(pd_dataframe_list)
        # save pandas.DataFrame to csv
        if save:
            output_dir = os.path.join(self.get_output_dir(), network_name, "cross-instance")
            data_file = os.path.join(output_dir, "data.csv")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            df.to_csv(data_file)
        return df

    @staticmethod
    def evaluate_network_cross_scene_keypoints(dcn, dataset, full_path_cross_instance_labels):
        """
        Evaluates the network on keypoint annotations across scenes
        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param full_path_cross_instance_labels:
        :type full_path_cross_instance_labels:
        :return:
        :rtype: pandas.DataFrame
        """

        utils.reset_random_seed()

        cross_instance_keypoint_labels = utils.getDictFromYamlFilename(full_path_cross_instance_labels)

        print "num cross instance labels", len(cross_instance_keypoint_labels)

        # Two-layer dict with:
        # - key:   the scene_name
        # - key:   the image_idx
        # - value: the descriptor image 
        descriptor_images = dict()

        # generate all descriptor images
        for keypoint_label in cross_instance_keypoint_labels:
            scene_name = keypoint_label["scene_name"]
            image_idx = keypoint_label["image_idx"]
            if scene_name not in descriptor_images:
                descriptor_images[scene_name] = dict()
            if image_idx in descriptor_images[scene_name]:
                continue
            rgb, _, _, _ = dataset.get_rgbd_mask_pose(scene_name, image_idx)
            rgb_tensor = dataset.rgb_image_to_tensor(rgb)
            res = dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()
            
            descriptor_images[scene_name][image_idx] = res


        pd_dataframe_list = []

        # generate all pairs of images
        counter = 0
        for subset in itertools.combinations(cross_instance_keypoint_labels, 2):
            counter += 1
            keypoint_data_a = subset[0]
            keypoint_data_b = subset[1]
            
            res_a = descriptor_images[keypoint_data_a["scene_name"]][keypoint_data_a["image_idx"]]
            res_b = descriptor_images[keypoint_data_b["scene_name"]][keypoint_data_b["image_idx"]]

            dataframe_list_temp = \
                DenseCorrespondenceEvaluation.single_image_pair_cross_scene_keypoints_quantitative_analysis(dcn, dataset, keypoint_data_a, keypoint_data_b, res_a, res_b)

            if dataframe_list_temp is None:
                print "no matches found, skipping"
                continue

            pd_dataframe_list += dataframe_list_temp


        print "num_pairs considered", counter
        df = pd.concat(pd_dataframe_list)

        return df

    @staticmethod
    def single_same_scene_image_pair_detection_analysis(dcn, dataset, scene_name,
                                                            img_a_data,
                                                            img_b_data,
                                                            num_matches,
                                                            debug):

        depth_a = np.asarray(img_a_data["depth"])
        depth_b = np.asarray(img_b_data["depth"])
        mask_a = np.asarray(img_a_data["mask"])
        mask_b = np.asarray(img_b_data["mask"])

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(img_a_data["rgb"])
        rgb_b_tensor = dataset.rgb_image_to_tensor(img_b_data["rgb"])

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        # find correspondences
        uv_a_vec, uv_b_vec, uv_a_not_detected_vec = correspondence_finder.batch_find_pixel_correspondences(depth_a, img_a_data["pose"],
                                                                            depth_b, img_b_data["pose"],
                                                                            img_a_mask=mask_a,
                                                                            K_a=img_a_data["K"],
                                                                            K_b=img_b_data["K"],
                                                                            num_attempts=200)


        if (uv_a_vec is None) or (len(uv_a_vec)==0):
            print "no matches found, returning"
            return None

        print len(uv_a_vec[0]), len(uv_b_vec[0]), len(uv_a_not_detected_vec[0])


        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []

        total_num_matches = len(uv_a_vec[0])
        num_matches = min(num_matches, total_num_matches)
        match_list = random.sample(range(0, total_num_matches), num_matches)

        if debug:
            match_list = [50]

        image_height, image_width = dcn.image_shape

        DCE = DenseCorrespondenceEvaluation

        for i in match_list:
            uv_a = (uv_a_vec[0][i], uv_a_vec[1][i])
            
            pd_template = DCE.compute_detection_statistics_true_positive(res_a, res_b, uv_a)
            pd_template.set_value('scene_name', scene_name)
            pd_template.set_value('img_a_idx', img_a_data["index"])
            pd_template.set_value('img_b_idx', img_b_data["index"])

            dataframe_list.append(pd_template.dataframe)

        total_num_not_detected = len(uv_a_not_detected_vec[0])
        num_not_detected = min(num_matches, total_num_not_detected)
        not_detected_list = random.sample(range(0,total_num_not_detected), num_not_detected)

        for i in not_detected_list:
            uv_a_not_detected = (uv_a_not_detected_vec[0][i], uv_a_not_detected_vec[1][i])

            pd_template = DCE.compute_detection_statistics_true_negative(res_a, res_b, uv_a_not_detected)
            pd_template.set_value('scene_name', scene_name)
            pd_template.set_value('img_a_idx', img_a_data["index"])
            pd_template.set_value('img_b_idx', img_b_data["index"])

            dataframe_list.append(pd_template.dataframe)

        return dataframe_list


    @staticmethod
    def compute_detection_statistics_true_positive(res_a, res_b, uv_a):
        uv_b_pred, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b)

        pd_template = DCNDetectionEvaluationPandaTemplate()
        pd_template.set_value('best_match_diff', best_match_diff)
        pd_template.set_value('true_positive', True)
        return pd_template

    @staticmethod
    def compute_detection_statistics_true_negative(res_a, res_b, uv_a):
        uv_b_pred, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b)

        pd_template = DCNDetectionEvaluationPandaTemplate()
        pd_template.set_value('best_match_diff', best_match_diff)
        pd_template.set_value('true_positive', False)
        return pd_template



    @staticmethod
    def evaluate_detection_on_network(dcn, dataset, num_image_pairs=25, num_matches_per_image_pair=100):
        """

        :param nn: A neural network DenseCorrespondenceNetwork
        :param test_dataset: DenseCorrespondenceDataset
            the dataset to draw samples from
        :return:
        """

        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation
        dcn.eval()

        logging_rate = 5


        pd_dataframe_list = []
        for i in xrange(0, num_image_pairs):

            # grab random scene
            scene_name = dataset.get_random_scene_name()
            if i % logging_rate == 0:
                print "computing statistics for image %d of %d, scene_name %s" %(i, num_image_pairs, scene_name)
                print "scene"

            if isinstance(dataset, DynamicSpartanDataset):
                img_a_data, img_b_data = dataset.get_img_pair_data(scene_name)
            else:
                print "found a static dataset"
                return NotImplementedError("need to implement get_img_pair_data")


            dataframe_list_temp =\
                DCE.single_same_scene_image_pair_detection_analysis(dcn, dataset, scene_name,
                                                            img_a_data,
                                                            img_b_data,
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
    def evaluate_network(dcn, dataset, num_image_pairs=25, num_matches_per_image_pair=100):
        """

        :param nn: A neural network DenseCorrespondenceNetwork
        :param test_dataset: DenseCorrespondenceDataset
            the dataset to draw samples from
        :return:
        """
        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation
        dcn.eval()

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
                                  mask_a=None, mask_b=None, plot_masked=False,descriptor_norm_type="mask_image"):
        """
        Plots the colormaps of descriptors for a pair of images
        :param res_a: descriptors for img_a
        :type res_a: numpy.ndarray
        :param res_b:
        :type res_b: numpy.ndarray
        :param descriptor_norm_type: what type of normalization to use for the
        full descriptor image
        :type : str
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
            res_a_norm = dc_plotting.normalize_descriptor(res_a, descriptor_image_stats[descriptor_norm_type])
            res_b_norm = dc_plotting.normalize_descriptor(res_b, descriptor_image_stats[descriptor_norm_type])


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
                res_a_norm_mask, res_b_norm_mask = dc_plotting.normalize_masked_descriptor_pair(res_a, res_b, mask_a, mask_b)
            else:
                res_a_norm_mask = dc_plotting.normalize_descriptor(res_a_mask, descriptor_image_stats['mask_image'])
                res_b_norm_mask = dc_plotting.normalize_descriptor(res_b_mask, descriptor_image_stats['mask_image'])

            res_a_norm_mask = res_a_norm_mask * mask_a_repeat
            res_b_norm_mask = res_b_norm_mask * mask_b_repeat

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

        # Loop over the labeled pixel matches once, before using different views
        # This lets us keep depth_a, depth_b, res_a, res_b without reloading
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
                                                        depth_b, mask_a, mask_b, uv_a, uv_b, pose_a,pose_b, res_a,
                                                        res_b, camera_intrinsics_matrix,
                                                        rgb_a=rgb_a, rgb_b=rgb_b, debug=False)

            pd_template.set_value('scene_name', scene_name_a+"+"+scene_name_b)
            pd_template.set_value('img_a_idx', int(img_a_idx))
            pd_template.set_value('img_b_idx', int(img_b_idx))

            dataframe_list.append(pd_template.dataframe)

        # Loop a second time over the labeled pixel matches
        # But this time try,
        #  for each I labeled pixel match pairs,
        #       for each J different views for image a, and
        #       for each K different views for image b
        # This will lead to I*J+I*K attempts at new pairs!
        # Could also do the cubic version...
        J = 10
        K = 10

        # Loop over labeled pixel matches
        for i in range(len(img_a_pixels)):
            uv_a = (img_a_pixels[i]["u"], img_a_pixels[i]["v"])
            uv_b = (img_b_pixels[i]["u"], img_b_pixels[i]["v"])
            uv_a = DCE.clip_pixel_to_image_size_and_round(uv_a, image_width, image_height)
            uv_b = DCE.clip_pixel_to_image_size_and_round(uv_b, image_width, image_height)

            # Loop over J different views for image a
            for j in range(J):
                different_view_a_idx = dataset.get_img_idx_with_different_pose(scene_name_a, pose_a, num_attempts=50)
                if different_view_a_idx is None:
                    logging.info("no frame with sufficiently different pose found, continuing")
                    continue
                diff_rgb_a, diff_depth_a, diff_mask_a, diff_pose_a = dataset.get_rgbd_mask_pose(scene_name_a, different_view_a_idx)
                diff_depth_a = np.asarray(diff_depth_a)
                diff_mask_a = np.asarray(diff_mask_a)
                (uv_a_vec, diff_uv_a_vec) = correspondence_finder.batch_find_pixel_correspondences(depth_a, pose_a, diff_depth_a, diff_pose_a,
                                                               uv_a=uv_a)
                if (uv_a_vec is None) or (len(uv_a_vec)==0):
                    logging.info("no matches found, continuing")
                    continue

                diff_rgb_a_tensor = dataset.rgb_image_to_tensor(diff_rgb_a)
                diff_res_a = dcn.forward_single_image_tensor(diff_rgb_a_tensor).data.cpu().numpy()

                diff_uv_a = (diff_uv_a_vec[0][0], diff_uv_a_vec[1][0])
                diff_uv_a = DCE.clip_pixel_to_image_size_and_round(diff_uv_a, image_width, image_height)

                pd_template = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(diff_depth_a,
                                                        depth_b, diff_mask_a, mask_b, diff_uv_a, uv_b, diff_pose_a, pose_b,
                                                        diff_res_a, res_b, camera_intrinsics_matrix,
                                                        rgb_a=diff_rgb_a, rgb_b=rgb_b, debug=False)
                pd_template.set_value('scene_name', scene_name_a+"+"+scene_name_b)
                pd_template.set_value('img_a_idx', int(different_view_a_idx))
                pd_template.set_value('img_b_idx', int(img_b_idx))

                dataframe_list.append(pd_template.dataframe)

            # Loop over K different views for image b
            for k in range(K):
                different_view_b_idx = dataset.get_img_idx_with_different_pose(scene_name_b, pose_b, num_attempts=50)
                if different_view_b_idx is None:
                    logging.info("no frame with sufficiently different pose found, continuing")
                    continue
                diff_rgb_b, diff_depth_b, diff_mask_b, diff_pose_b = dataset.get_rgbd_mask_pose(scene_name_b, different_view_b_idx)
                diff_depth_b = np.asarray(diff_depth_b)
                diff_mask_b = np.asarray(diff_mask_b)
                (uv_b_vec, diff_uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(depth_b, pose_b, diff_depth_b, diff_pose_b,
                                                               uv_a=uv_b)
                if uv_b_vec is None:
                    logging.info("no matches found, continuing")
                    continue

                diff_rgb_b_tensor = dataset.rgb_image_to_tensor(diff_rgb_b)
                diff_res_b = dcn.forward_single_image_tensor(diff_rgb_b_tensor).data.cpu().numpy()

                diff_uv_b = (diff_uv_b_vec[0][0], diff_uv_b_vec[1][0])
                diff_uv_b = DCE.clip_pixel_to_image_size_and_round(diff_uv_b, image_width, image_height)

                pd_template = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(depth_a,
                                                        diff_depth_b, mask_a, diff_mask_b, uv_a, diff_uv_b, pose_a, diff_pose_b,
                                                        res_a, diff_res_b, camera_intrinsics_matrix,
                                                        rgb_a=rgb_a, rgb_b=diff_rgb_b, debug=False)
                pd_template.set_value('scene_name', scene_name_a+"+"+scene_name_b)
                pd_template.set_value('img_a_idx', int(img_a_idx))
                pd_template.set_value('img_b_idx', int(different_view_b_idx))

                dataframe_list.append(pd_template.dataframe)

        return dataframe_list

    @staticmethod
    def single_across_object_image_pair_quantitative_analysis(dcn, dataset, scene_name_a, scene_name_b,
                                                img_a_idx, img_b_idx, object_id_a, object_id_b, 
                                                num_uv_a_samples=100,
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


        rgb_a, depth_a, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)
        rgb_b, depth_b, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

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

        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []

        logging_rate = 100

        image_height, image_width = dcn.image_shape

        DCE = DenseCorrespondenceEvaluation

        sampled_idx_list = random_sample_from_masked_image(mask_a, num_uv_a_samples)
        # If the list is empty, return an empty list
        if len(sampled_idx_list) == 0:
            return dataframe_list

        for i in range(num_uv_a_samples):

            uv_a = [sampled_idx_list[1][i], sampled_idx_list[0][i]]

            pd_template = DCE.compute_descriptor_match_statistics_no_ground_truth(uv_a, res_a,
                                                                  res_b,
                                                                  rgb_a=rgb_a,
                                                                  rgb_b=rgb_b,
                                                                  depth_a=depth_a,
                                                                  depth_b=depth_b,
                                                                  debug=debug)

            pd_template.set_value('scene_name_a', scene_name_a)
            pd_template.set_value('scene_name_b', scene_name_b)
            pd_template.set_value('object_id_a', object_id_a)
            pd_template.set_value('object_id_b', object_id_b)
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
        :return: List of pandas DataFrame objects
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

        if (uv_a_vec is None) or (len(uv_a_vec)==0):
            print "no matches found, returning"
            return None

        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []

        total_num_matches = len(uv_a_vec[0])
        num_matches = min(num_matches, total_num_matches)
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
                                                                  mask_a,
                                                                  mask_b,
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
    def compute_descriptor_match_statistics_no_ground_truth(uv_a, res_a, res_b, rgb_a=None, rgb_b=None,
                                                            depth_a=None, depth_b=None, debug=False):
        """
        Computes statistics of descriptor pixelwise match when there is zero ground truth data.

        :param res_a: descriptor for image a, of shape (H,W,D)
        :type res_a: numpy array
        :param res_b: descriptor for image b, of shape (H,W,D)
        :param debug: whether or not to print visualization
        :type debug:
        """

        DCE = DenseCorrespondenceEvaluation

        # compute best match
        uv_b, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b, debug=debug)

        if debug:
            correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b,
                                                               uv_a, uv_b, show=True)


        pd_template = DCNEvaluationPandaTemplateAcrossObject()
        pd_template.set_value('norm_diff_descriptor_best_match', best_match_diff)
        return pd_template


    @staticmethod
    def compute_descriptor_match_statistics(depth_a, depth_b, mask_a, mask_b, uv_a, uv_b, pose_a, pose_b,
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
        :return: Dense
        :rtype: DCNEvaluationPandaTemplate
        """

        DCE = DenseCorrespondenceEvaluation

        # compute best match
        uv_b_pred, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b, debug=debug)

        # norm_diffs shape is (H,W)

        # compute best match on mask only
        mask_b_inv = 1-mask_b
        masked_norm_diffs = norm_diffs + mask_b_inv*1e6

        best_match_flattened_idx_masked = np.argmin(masked_norm_diffs)
        best_match_xy_masked = np.unravel_index(best_match_flattened_idx_masked, masked_norm_diffs.shape)
        best_match_diff_masked = masked_norm_diffs[best_match_xy_masked]
        uv_b_pred_masked = (best_match_xy_masked[1], best_match_xy_masked[0])

        # compute pixel space difference
        pixel_match_error_l2 = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=2)
        pixel_match_error_l2_masked = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred_masked)), ord=2)
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

        (v_indices_better_than_ground_truth_masked, u_indices_better_than_ground_truth_masked) = np.where(masked_norm_diffs < norm_diff_descriptor_ground_truth)
        num_pixels_closer_than_ground_truth_masked = len(u_indices_better_than_ground_truth_masked)
        num_pixels_in_masked_image = len(np.nonzero(mask_b)[0])
        fraction_pixels_closer_than_ground_truth_masked = num_pixels_closer_than_ground_truth_masked*1.0/num_pixels_in_masked_image

        # new metric: average l2 distance of the pixels better than ground truth
        if num_pixels_closer_than_ground_truth == 0:
            average_l2_distance_for_false_positives = 0.0
        else:
            l2_distances = np.sqrt((u_indices_better_than_ground_truth - uv_b[0])**2 + (v_indices_better_than_ground_truth - uv_b[1])**2)
            average_l2_distance_for_false_positives = np.average(l2_distances)

        # new metric: average l2 distance of the pixels better than ground truth
        if num_pixels_closer_than_ground_truth_masked == 0:
            average_l2_distance_for_false_positives_masked = 0.0
        else:
            l2_distances_masked = np.sqrt((u_indices_better_than_ground_truth_masked - uv_b[0])**2 + (v_indices_better_than_ground_truth_masked - uv_b[1])**2)
            average_l2_distance_for_false_positives_masked = np.average(l2_distances_masked)

        # extract depth values, note the indexing order of u,v has to be reversed
        uv_a_depth = depth_a[uv_a[1], uv_a[0]] / DEPTH_IM_SCALE # check if this is not None
        uv_b_depth = depth_b[uv_b[1], uv_b[0]] / DEPTH_IM_SCALE
        uv_b_pred_depth = depth_b[uv_b_pred[1], uv_b_pred[0]] / DEPTH_IM_SCALE
        uv_b_pred_depth_is_valid = DenseCorrespondenceEvaluation.is_depth_valid(uv_b_pred_depth)
        uv_b_pred_depth_masked = depth_b[uv_b_pred_masked[1], uv_b_pred_masked[0]] / DEPTH_IM_SCALE
        uv_b_pred_depth_is_valid_masked = DenseCorrespondenceEvaluation.is_depth_valid(uv_b_pred_depth_masked)
        is_valid = uv_b_pred_depth_is_valid
        is_valid_masked = uv_b_pred_depth_is_valid_masked

        uv_a_pos = DCE.compute_3d_position(uv_a, uv_a_depth, camera_matrix, pose_a)
        uv_b_pos = DCE.compute_3d_position(uv_b, uv_b_depth, camera_matrix, pose_b)
        uv_b_pred_pos = DCE.compute_3d_position(uv_b_pred, uv_b_pred_depth, camera_matrix, pose_b)
        uv_b_pred_pos_masked = DCE.compute_3d_position(uv_b_pred_masked, uv_b_pred_depth_masked, camera_matrix, pose_b)

        diff_ground_truth_3d = uv_b_pos - uv_a_pos

        diff_pred_3d = uv_b_pos - uv_b_pred_pos
        diff_pred_3d_masked = uv_b_pos - uv_b_pred_pos_masked

        """
        We need to decide how to treat the "3D" error for a pixel
        that we don't have depth for.
        """
        # Option 1, if np.nan, then this doesn't get reflected in the metric
        #NO_DEPTH_3D_ERROR = np.nan
        
        # Option 2, if some large value, then this will show up on the far right of the plot
        NO_DEPTH_3D_ERROR = 1.0 # unit is in meters

        if DCE.is_depth_valid(uv_b_depth):
            norm_diff_ground_truth_3d = np.linalg.norm(diff_ground_truth_3d)
        else:
            norm_diff_ground_truth_3d = NO_DEPTH_3D_ERROR

        if DCE.is_depth_valid(uv_b_depth) and DCE.is_depth_valid(uv_b_pred_depth):
            norm_diff_pred_3d = np.linalg.norm(diff_pred_3d)
        else:
            norm_diff_pred_3d = NO_DEPTH_3D_ERROR

        if DCE.is_depth_valid(uv_b_depth) and DCE.is_depth_valid(uv_b_pred_depth_masked):
            norm_diff_pred_3d_masked = np.linalg.norm(diff_pred_3d_masked)
        else:
            norm_diff_pred_3d_masked = NO_DEPTH_3D_ERROR

        if debug:

            fig, axes = correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b,
                                                               uv_a, uv_b, show=False)

            correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b,
                                                               uv_a, uv_b_pred,
                                                               use_previous_plot=(fig, axes),
                                                               show=True,
                                                               circ_color='purple')

        pd_template = DCNEvaluationPandaTemplate()
        pd_template.set_value('norm_diff_descriptor', best_match_diff)
        pd_template.set_value('norm_diff_descriptor_masked', best_match_diff_masked)
        pd_template.set_value('is_valid', is_valid)
        pd_template.set_value('is_valid_masked', is_valid_masked)

        pd_template.set_value('norm_diff_ground_truth_3d', norm_diff_ground_truth_3d)
        pd_template.set_value('norm_diff_pred_3d', norm_diff_pred_3d)
        pd_template.set_value('norm_diff_pred_3d_masked', norm_diff_pred_3d_masked)
        

        pd_template.set_value('norm_diff_descriptor_ground_truth', norm_diff_descriptor_ground_truth)

        pd_template.set_value('pixel_match_error_l2', pixel_match_error_l2)
        pd_template.set_value('pixel_match_error_l2_masked', pixel_match_error_l2_masked)
        pd_template.set_value('pixel_match_error_l1', pixel_match_error_l1)

        pd_template.set_value('fraction_pixels_closer_than_ground_truth', fraction_pixels_closer_than_ground_truth)
        pd_template.set_value('fraction_pixels_closer_than_ground_truth_masked', fraction_pixels_closer_than_ground_truth_masked)
        pd_template.set_value('average_l2_distance_for_false_positives', average_l2_distance_for_false_positives)
        pd_template.set_value('average_l2_distance_for_false_positives_masked', average_l2_distance_for_false_positives_masked)


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

        # DYNAMIC 
        # rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name, 0, img_a_idx)
        # rgb_b, _, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name, 1, img_a_idx)

        # STATIC
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
    def single_image_pair_keypoint_qualitative_analysis(dcn, dataset, keypoint_data_a,
                                                        keypoint_data_b,
                                                        heatmap_kernel_variance=0.25,
                                                        blend_weight_original_image=0.3,
                                                        plot_title="Keypoints"):
        """
        Wrapper for qualitative analysis of a pair of images using keypoint annotations
        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param keypoint_data_a: pandas Series
        :type keypoint_data_a:
        :param keypoint_data_b:
        :type keypoint_data_b:
        :return:
        :rtype:
        """
        DCE = DenseCorrespondenceEvaluation

        image_height, image_width = dcn.image_shape

        scene_name_a = keypoint_data_a['scene_name']
        img_a_idx = keypoint_data_a['image_idx']
        uv_a = (keypoint_data_a['u'], keypoint_data_a['v'])
        uv_a = DCE.clip_pixel_to_image_size_and_round(uv_a, image_width, image_height)

        scene_name_b = keypoint_data_b['scene_name']
        img_b_idx = keypoint_data_b['image_idx']
        uv_b = (keypoint_data_b['u'], keypoint_data_b['v'])
        uv_b = DCE.clip_pixel_to_image_size_and_round(uv_b, image_width, image_height)


        rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)

        rgb_b, _, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        best_match_uv, best_match_diff, norm_diffs = \
        DenseCorrespondenceNetwork.find_best_match(uv_a, res_a, res_b, debug=False)



        # visualize image and then heatmap
        diam = 0.03
        dist = 0.01
        kp1 = []
        kp2 = []
        kp1.append(cv2.KeyPoint(uv_a[0], uv_a[1], diam))
        kp2.append(cv2.KeyPoint(best_match_uv[0], best_match_uv[1], diam))


        matches = [] # list of cv2.DMatch
        matches.append(cv2.DMatch(0,0,dist))

        gray_a_numpy = cv2.cvtColor(np.asarray(rgb_a), cv2.COLOR_BGR2GRAY)
        gray_b_numpy = cv2.cvtColor(np.asarray(rgb_b), cv2.COLOR_BGR2GRAY)
        img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, matches, flags=2, outImg=gray_b_numpy, matchColor=(255,0,0))

        fig, axes = plt.subplots(nrows=2, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        axes[0].imshow(img3)
        axes[0].set_title(plot_title)

        # visualize the heatmap
        heatmap_color = vis_utils.compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, heatmap_kernel_variance)

        # convert heatmap to RGB (it's in BGR now)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


        alpha = blend_weight_original_image
        beta = 1-alpha
        blended = cv2.addWeighted(np.asarray(rgb_b), alpha, heatmap_color_rgb, beta, 0)

        axes[1].imshow(blended)
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
        # TODO: if this mask is empty, this function will not be happy
        # de-prioritizing since this is only for qualitative evaluation plots
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
                DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a, res_b)

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
    def single_image_pair_cross_scene_keypoints_quantitative_analysis(dcn, dataset, keypoint_data_a,
                                                                      keypoint_data_b, res_a=None, res_b=None):
        """
        Quantitative analysis of cross instance keypoint annotations. This is used in the
        class consistent setting

        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param keypoint_data_a:
        :type keypoint_data_a:
        :param keypoint_data_b:
        :type keypoint_data_b:
        :return: List of pandas DataFrame objects
        :rtype:
        """

        DCE = DenseCorrespondenceEvaluation

        scene_name_a = keypoint_data_a['scene_name']
        object_id_a = keypoint_data_a['object_id']
        img_a_idx = keypoint_data_a['image_idx']

        scene_name_b = keypoint_data_b['scene_name']
        object_id_b = keypoint_data_b['object_id']
        img_b_idx = keypoint_data_b['image_idx']

        rgb_a, depth_a, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)

        rgb_b, depth_b, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

        depth_a = np.asarray(depth_a)
        depth_b = np.asarray(depth_b)
        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        if res_a is None and res_b is None:
            # compute dense descriptors
            rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
            rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

            # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
            res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
            res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()


        # vectors to allow re-ordering
        rgb = [rgb_a, rgb_b]
        depth = [depth_a, depth_b]
        mask = [mask_a, mask_b]
        scene_name = [scene_name_a, scene_name_b]
        img_idx = [img_a_idx, img_b_idx]
        pose = [pose_a, pose_b]
        res = [res_a, res_b]
        object_id = [object_id_a, object_id_b]

        camera_intrinsics_a = dataset.get_camera_intrinsics(scene_name_a)
        camera_intrinsics_b = dataset.get_camera_intrinsics(scene_name_b)
        if not np.allclose(camera_intrinsics_a.K, camera_intrinsics_b.K):
            print "Currently cannot handle two different camera K matrices in different scenes!"
            print "But you could add this..."
        camera_intrinsics_matrix = camera_intrinsics_a.K

        image_height, image_width = dcn.image_shape
        DCE = DenseCorrespondenceEvaluation
        dataframe_list = []

        ordering = ["standard", "reverse"]

        for kp_name, data_a in keypoint_data_a['keypoints'].iteritems():
            if kp_name not in keypoint_data_b['keypoints']:
                raise ValueError("keypoint %s appears in one list of annotated data but"
                                 "not the other" %(kp_name))

            data_b = keypoint_data_b['keypoints'][kp_name]

            data = [data_a, data_b]

            for order in ordering:
                if order == "standard":
                    idx_1 = 0
                    idx_2 = 1
                elif order == "reverse":
                    idx_1 = 1
                    idx_2 = 0
                else:
                    raise ValueError("you should never get here")


                uv_1 = DCE.clip_pixel_to_image_size_and_round((data[idx_1]['u'], data[idx_2]['v']), image_width, image_height)
                uv_2 = DCE.clip_pixel_to_image_size_and_round((data[idx_2]['u'], data[idx_2]['v']), image_width,
                                                              image_height)

                pd_template = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(depth[idx_1],
                                                                                                depth[idx_2],
                                                                                                mask[idx_1],
                                                                                                mask[idx_2],
                                                                                                uv_1,
                                                                                                uv_2,
                                                                                                pose[idx_1],
                                                                                                pose[idx_2],
                                                                                                res[idx_1],
                                                                                                res[idx_2],
                                                                                                camera_intrinsics_matrix,
                                                                                                rgb_a=rgb[idx_1], rgb_b=rgb[idx_2],
                                                                                                debug=False)

                pd_template.set_value('img_a_idx', img_idx[idx_1])
                pd_template.set_value('img_b_idx', img_idx[idx_2])
                pd_template.set_value('scene_name_a', scene_name[idx_1])
                pd_template.set_value('scene_name_b', scene_name[idx_2])
                pd_template.set_value('object_id_a', object_id[idx_1])
                pd_template.set_value('object_id_b', object_id[idx_2])
                pd_template.set_value('keypoint_name', kp_name)


                dataframe_list.append(pd_template.dataframe)

        return dataframe_list

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
    def parse_cross_scene_data(dataset):
        """
        This takes a dataset.config, and concatenates together
        a list of all of the cross scene data annotated pairs.
        """
        evaluation_labeled_data_paths = []

        # add the multi object list
        # Note: (manuelli) why is this treated differently than the single object
        # case?
        evaluation_labeled_data_paths += dataset.config["multi_object"]["evaluation_labeled_data_path"]
        
        # add all of the single object lists
        for object_key, val in dataset.config["single_object"].iteritems():
            if "evaluation_labeled_data_path" in val:
                evaluation_labeled_data_paths += val["evaluation_labeled_data_path"]

        if len(evaluation_labeled_data_paths) == 0:
            print "Could not find labeled cross scene data for this dataset."
            print "It needs to be set in the dataset.yaml of the folder from which"
            print "this network is loaded from."
            return

        cross_scene_data = []

        for path in evaluation_labeled_data_paths:
            cross_scene_data_full_path = utils.convert_data_relative_path_to_absolute_path(path, assert_path_exists=True)
            this_cross_scene_data = utils.getDictFromYamlFilename(cross_scene_data_full_path)
            cross_scene_data += this_cross_scene_data

        return cross_scene_data

    @staticmethod
    def evaluate_network_qualitative_cross_scene(dcn, dataset, draw_human_annotations=True):
        """
        This will search for the "evaluation_labeled_data_path" in the dataset.yaml,
        and use pairs of images that have been human-labeled across scenes.
        """

        dcn.eval()

        cross_scene_data = DenseCorrespondenceEvaluation.parse_cross_scene_data(dataset)
        
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
    def get_random_image_pairs(dataset):
        """
        Given a dataset, chose a random scene, and a handful of image pairs from
        that scene.

        :param dataset: dataset from which to draw a scene and image pairs
        :type dataset: SpartanDataset

        :return: scene_name, img_pairs
        :rtype: str, list of lists, where each of the lists are [img_a_idx, img_b_idx], for example:
            [[113,220],
             [114,225]]
        """
        scene_name = dataset.get_random_scene_name()
        img_pairs = []
        for _ in range(5):
            img_a_idx  = dataset.get_random_image_index(scene_name)
            pose_a     = dataset.get_pose_from_scene_name_and_idx(scene_name, img_a_idx)
            img_b_idx  = dataset.get_img_idx_with_different_pose(scene_name, pose_a, num_attempts=100)
            if img_b_idx is None:
                continue
            img_pairs.append([img_a_idx, img_b_idx])
        return scene_name, img_pairs

    @staticmethod
    def get_random_scenes_and_image_pairs(dataset):
        """
        Given a dataset, chose a variety of random scenes and image pairs

        :param dataset: dataset from which to draw a scene and image pairs
        :type dataset: SpartanDataset

        :return: scene_names, img_pairs
        :rtype: list[str], list of lists, where each of the lists are [img_a_idx, img_b_idx], for example:
            [[113,220],
             [114,225]]
        """

        scene_names = []

        img_pairs = []
        for _ in range(5):
            scene_name = dataset.get_random_scene_name()
            
            img_a_idx = dataset.get_random_image_index(scene_name)

            # FOR STATIC
            pose_a = dataset.get_pose_from_scene_name_and_idx(scene_name, img_a_idx)
            img_b_idx = dataset.get_img_idx_with_different_pose(scene_name, pose_a, num_attempts=100)
            if img_b_idx is None:
                continue
            img_pairs.append([img_a_idx, img_b_idx])
            
            # FOR DYNAMIC
            #img_pairs.append([img_a_idx, img_a_idx])

            scene_names.append(scene_name)

        return scene_names, img_pairs

    @staticmethod
    def evaluate_network_qualitative(dcn, dataset, num_image_pairs=5, randomize=False,
                                     scene_type=None):

        dcn.eval()
        # Train Data
        print "\n\n-----------Train Data Evaluation----------------"
        if randomize:
            scene_names, img_pairs = DenseCorrespondenceEvaluation.get_random_scenes_and_image_pairs(dataset)
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

            scene_names = [scene_name]*len(img_pairs)

        for scene_name, img_pair in zip(scene_names, img_pairs):
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
            scene_names, img_pairs = DenseCorrespondenceEvaluation.get_random_scenes_and_image_pairs(dataset)
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

            scene_names = [scene_name] * len(img_pairs)


        for scene_name, img_pair in zip(scene_names, img_pairs):
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
                scene_name, img_pairs = DenseCorrespondenceEvaluation.get_random_image_pairs(dataset)
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
    def compute_loss_on_dataset(dcn, data_loader, loss_config, num_iterations=500,):
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
        dcn.eval()

        # loss_vec = np.zeros(num_iterations)
        loss_vec = []
        match_loss_vec = []
        non_match_loss_vec = []
        counter = 0
        pixelwise_contrastive_loss = PixelwiseContrastiveLoss(dcn.image_shape, config=loss_config)

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

        utils.reset_random_seed()

        dcn.eval()
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

            
            mask_flat = mask_tensor.view(-1,1).squeeze(1)

            # now do the same for the masked image
            # gracefully handle the case where the mask is all zeros
            mask_indices_flat = torch.nonzero(mask_flat)
            if len(mask_indices_flat) == 0:
                return None, None     

            mask_indices_flat = mask_indices_flat.squeeze(1)
            
                
            # print "mask_flat.shape", mask_flat.shape

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


            # handles the case of an empty mask
            if mask_image_stats is None:
                logging.info("Mask was empty, skipping")
                continue


            update_stats(stats['entire_image'], entire_image_stats)
            update_stats(stats['mask_image'], mask_image_stats)


        for key, val in stats.iteritems():
            val['mean'] = 1.0/num_images * val['mean']
            for field in val:
                val[field] = val[field].tolist()

        if save_to_file:
            if filename is None:
                path_to_params_folder = dcn.config['path_to_network_params_folder']
                path_to_params_folder = utils.convert_data_relative_path_to_absolute_path(path_to_params_folder)
                filename = os.path.join(path_to_params_folder, 'descriptor_statistics.yaml')

            utils.saveToYaml(stats, filename)



        return stats

    @staticmethod
    def run_detection_evaluation_on_network(model_folder, num_image_pairs=100,
                                  num_matches_per_image_pair=100,
                                  save_folder_name="analysis",
                                  compute_descriptor_statistics=True, 
                                  cross_scene=True,
                                  dataset=None,
                                  iteration=None):
        """
        Runs detection quantitative evaluations on the model folder
        """

        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation

        model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder, assert_path_exists=True)

        print "SETTING GLOBAL FOR MATCH TYPE"
        training_dict = yaml.load(file(os.path.join(model_folder, "training.yaml")))
        compute_best_match_with =  training_dict["dense_correspondence_network"]["compute_best_match_with"]
        utils.add_dense_correspondence_to_python_path()
        import dense_correspondence.network.dense_correspondence_network
        dense_correspondence.network.dense_correspondence_network.COMPUTE_BEST_MATCH_WITH = compute_best_match_with


        # save it to a csv file
        output_dir = os.path.join(model_folder, save_folder_name)
        train_output_dir = os.path.join(output_dir, "detection_train")
        test_output_dir = os.path.join(output_dir, "detection_test")

        # create the necessary directories
        for dir in [output_dir, train_output_dir, test_output_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)

        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, iteration=iteration)
        dcn.eval()

        if dataset is None:
            dataset = dcn.load_training_dataset()

        # evaluate on training data and on test data
        logging.info("Evaluating detection on train data")
        dataset.set_train_mode()
        pd_dataframe_list, df = DCE.evaluate_detection_on_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        detection_train_csv = os.path.join(train_output_dir, "data.csv")
        df.to_csv(detection_train_csv)

        logging.info("Evaluating detection on test data")
        dataset.set_test_mode()
        pd_dataframe_list, df = DCE.evaluate_detection_on_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        detection_test_csv = os.path.join(test_output_dir, "data.csv")
        df.to_csv(detection_test_csv)

        logging.info("Finished running evaluation on network")


    @staticmethod
    def run_evaluation_on_network(model_folder, num_image_pairs=100,
                                  num_matches_per_image_pair=100,
                                  save_folder_name="analysis",
                                  compute_descriptor_statistics=True, 
                                  cross_scene=True,
                                  dataset=None,
                                  iteration=None):
        """
        Runs all the quantitative evaluations on the model folder
        Creates a folder model_folder/analysis that stores the information.

        Performs several steps:

        1. compute dataset descriptor stats
        2. compute quantitative eval csv files
        3. make quantitative plots, save as a png for easy viewing


        :param model_folder:
        :type model_folder:
        :return:
        :rtype:
        """

        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation

        model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder, assert_path_exists=True)

        # save it to a csv file
        output_dir = os.path.join(model_folder, save_folder_name)
        train_output_dir = os.path.join(output_dir, "train")
        test_output_dir = os.path.join(output_dir, "test")
        cross_scene_output_dir = os.path.join(output_dir, "cross_scene")

        # create the necessary directories
        for dir in [output_dir, train_output_dir, test_output_dir, cross_scene_output_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)


        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, iteration=iteration)
        dcn.eval()

        if dataset is None:
            dataset = dcn.load_training_dataset()

        # compute dataset statistics
        if compute_descriptor_statistics:
            logging.info("Computing descriptor statistics on dataset")
            DCE.compute_descriptor_statistics_on_dataset(dcn, dataset, num_images=100, save_to_file=True)


        # evaluate on training data and on test data
        logging.info("Evaluating network on train data")
        dataset.set_train_mode()
        pd_dataframe_list, df = DCE.evaluate_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        train_csv = os.path.join(train_output_dir, "data.csv")
        df.to_csv(train_csv)

        logging.info("Evaluating network on test data")
        dataset.set_test_mode()
        pd_dataframe_list, df = DCE.evaluate_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        test_csv = os.path.join(test_output_dir, "data.csv")
        df.to_csv(test_csv)


        if cross_scene:
            logging.info("Evaluating network on cross scene data")
            df = DCE.evaluate_network_cross_scene(dcn=dcn, dataset=dataset, save=False)
            cross_scene_csv = os.path.join(cross_scene_output_dir, "data.csv")
            df.to_csv(cross_scene_csv)

        logging.info("Making plots")
        DCEP = DenseCorrespondenceEvaluationPlotter
        fig_axes = DCEP.run_on_single_dataframe(train_csv, label="train", save=False)
        fig_axes = DCEP.run_on_single_dataframe(test_csv, label="test", save=False, previous_fig_axes=fig_axes)
        if cross_scene:
            fig_axes = DCEP.run_on_single_dataframe(cross_scene_csv, label="cross_scene", save=False,
                                                    previous_fig_axes=fig_axes)

        fig, _ = fig_axes        
        save_fig_file = os.path.join(output_dir, "quant_plots.png")
        fig.savefig(save_fig_file)

        # only do across object analysis if have multiple single objects
        if dataset.get_number_of_unique_single_objects() > 1:
            across_object_output_dir = os.path.join(output_dir, "across_object")
            if not os.path.isdir(across_object_output_dir):
                os.makedirs(across_object_output_dir)
            logging.info("Evaluating network on across object data")
            df = DCE.evaluate_network_across_objects(dcn=dcn, dataset=dataset)
            across_object_csv = os.path.join(across_object_output_dir, "data.csv")
            df.to_csv(across_object_csv)
            DCEP.run_on_single_dataframe_across_objects(across_object_csv, label="across_object", save=True)


        logging.info("Finished running evaluation on network")

    @staticmethod
    def run_cross_instance_keypoint_evaluation_on_network(model_folder, path_to_cross_instance_labels,
                                  save_folder_name="analysis/cross_scene_keypoints",
                                  dataset=None, save=True):
        """
        Runs cross instance keypoint evaluation on the given network.
        Creates a folder model_folder/<save_folder_name> that stores the information.

        :param model_folder: folder where trained network is
        :type model_folder:
        :param path_to_cross_instance_labels: path to location of cross instance data. Can be full path
        or path relative to the data directory
        :type path_to_cross_instance_labels: str
        :param save_folder_name: place where data is being saved
        :type save_folder_name: str
        :param dataset: SpartanDataset
        :type dataset:
        :return:
        :rtype:
        """
        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation

        model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder, assert_path_exists=True)



        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder)
        dcn.eval()

        if dataset is None:
            dataset = dcn.load_training_dataset()

        path_to_cross_instance_labels = utils.convert_data_relative_path_to_absolute_path(path_to_cross_instance_labels, assert_path_exists=True)
        df = DCE.evaluate_network_cross_scene_keypoints(dcn, dataset, path_to_cross_instance_labels)


        if save:
            # save it to a csv file
            output_dir = os.path.join(model_folder, save_folder_name)
            # create the necessary directories
            for dir in [output_dir]:
                if not os.path.isdir(dir):
                    os.makedirs(dir)

            save_filename = os.path.join(output_dir, 'data.csv')
            df.to_csv(save_filename)

        logging.info("Finished running cross scene keypoint evaluation")
        return df


    @staticmethod
    def make_2d_cluster_plot(dcn, dataset, plot_background=False):
        """
        This function randomly samples many points off of different objects and the background,
        and makes an object-labeled scatter plot of where these descriptors are.
        """

        print "Checking to make sure this is a 2D or 3D descriptor"
        print "If you'd like you could add projection methods for higher dimension descriptors"
        assert ((dcn.descriptor_dimension == 2) or (dcn.descriptor_dimension == 3))

        if dcn.descriptor_dimension == 3:
            use_3d = True
            d = 3
            print "This descriptor_dimension is 3d"
            print "I'm going to make 3 plots for you: xy, yz, xz"
        else:
            use_3d = False
            d = 2

        # randomly grab object ID, and scene

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        descriptors_known_objects_samples = dict()
        if use_3d:
            descriptors_known_objects_samples_xy = dict()
            descriptors_known_objects_samples_yz = dict()
            descriptors_known_objects_samples_xz = dict()

        descriptors_background_samples = np.zeros((0,d))

        if use_3d:
            descriptors_background_samples_xy = np.zeros((0,2))
            descriptors_background_samples_yz = np.zeros((0,2))
            descriptors_background_samples_xz = np.zeros((0,2))

        num_objects = dataset.get_number_of_unique_single_objects()
        num_samples_per_image = 100

        for i in range(100):
            object_id, object_id_int = dataset.get_random_object_id_and_int()

            scene_name = dataset.get_random_single_object_scene_name(object_id)
            img_idx = dataset.get_random_image_index(scene_name)
            rgb = dataset.get_rgb_image_from_scene_name_and_idx(scene_name, img_idx)
            mask = dataset.get_mask_image_from_scene_name_and_idx(scene_name, img_idx)

            mask_torch = torch.from_numpy(np.asarray(mask)).long()
            mask_inv = 1 - mask_torch

            object_uv_samples     = correspondence_finder.random_sample_from_masked_image_torch(mask_torch, num_samples_per_image)
            background_uv_samples = correspondence_finder.random_sample_from_masked_image_torch(mask_inv, num_samples_per_image/num_objects)

            object_u_samples = object_uv_samples[0].numpy()
            object_v_samples = object_uv_samples[1].numpy()

            background_u_samples = background_uv_samples[0].numpy()
            background_v_samples = background_uv_samples[1].numpy()

            # This snippet will plot where the samples are coming from in the image
            # plt.scatter(object_u_samples, object_v_samples, c="g", alpha=0.5, label="object")
            # plt.scatter(background_u_samples, background_v_samples, c="k", alpha=0.5, label="background")
            # plt.legend()
            # plt.show()

            img_tensor = dataset.rgb_image_to_tensor(rgb)
            res = dcn.forward_single_image_tensor(img_tensor)  # [H, W, D]
            res = res.data.cpu().numpy()

            descriptors_object = np.zeros((len(object_u_samples),d))
            for j in range(len(object_u_samples)):
                descriptors_object[j,:] = res[object_v_samples[j], object_u_samples[j], :]
            if use_3d:
                descriptors_object_xy = np.zeros((len(object_u_samples),2))
                descriptors_object_yz = np.zeros((len(object_u_samples),2))
                descriptors_object_xz = np.zeros((len(object_u_samples),2))
                for j in range(len(object_u_samples)):
                    descriptors_object_xy[j,:] = res[object_v_samples[j], object_u_samples[j], 0:2]
                    descriptors_object_yz[j,:] = res[object_v_samples[j], object_u_samples[j], 1:3]
                    descriptors_object_xz[j,:] = res[object_v_samples[j], object_u_samples[j], 0::2]

            descriptors_background = np.zeros((len(background_u_samples),d))
            for j in range(len(background_u_samples)):
                descriptors_background[j,:] = res[background_v_samples[j], background_u_samples[j], :]
            if use_3d:
                descriptors_background_xy = np.zeros((len(background_u_samples),2))
                descriptors_background_yz = np.zeros((len(background_u_samples),2))
                descriptors_background_xz = np.zeros((len(background_u_samples),2))
                for j in range(len(background_u_samples)):
                    descriptors_background_xy[j,:] = res[background_v_samples[j], background_u_samples[j], 0:2]
                    descriptors_background_yz[j,:] = res[background_v_samples[j], background_u_samples[j], 1:3]
                    descriptors_background_xz[j,:] = res[background_v_samples[j], background_u_samples[j], 0::2]


            # This snippet will plot the descriptors just from this image
            # plt.scatter(descriptors_object[:,0], descriptors_object[:,1], c="g", alpha=0.5, label=object_id)
            # plt.scatter(descriptors_background[:,0], descriptors_background[:,1], c="k", alpha=0.5, label="background")
            # plt.legend()
            # plt.show()

            if object_id not in descriptors_known_objects_samples:
                descriptors_known_objects_samples[object_id] = descriptors_object
                if use_3d:
                    descriptors_known_objects_samples_xy[object_id] = descriptors_object_xy
                    descriptors_known_objects_samples_yz[object_id] = descriptors_object_yz
                    descriptors_known_objects_samples_xz[object_id] = descriptors_object_xz
            else:
                descriptors_known_objects_samples[object_id] = np.vstack((descriptors_known_objects_samples[object_id], descriptors_object))
                if use_3d:
                    descriptors_known_objects_samples_xy[object_id] = np.vstack((descriptors_known_objects_samples_xy[object_id], descriptors_object_xy))
                    descriptors_known_objects_samples_yz[object_id] = np.vstack((descriptors_known_objects_samples_yz[object_id], descriptors_object_yz))
                    descriptors_known_objects_samples_xz[object_id] = np.vstack((descriptors_known_objects_samples_xz[object_id], descriptors_object_xz))

            descriptors_background_samples = np.vstack((descriptors_background_samples, descriptors_background))
            if use_3d:
                descriptors_background_samples_xy = np.vstack((descriptors_background_samples_xy, descriptors_background_xy))
                descriptors_background_samples_yz = np.vstack((descriptors_background_samples_yz, descriptors_background_yz))
                descriptors_background_samples_xz = np.vstack((descriptors_background_samples_xz, descriptors_background_xz))



        print "ALL"
        if not use_3d:
            for key, value in descriptors_known_objects_samples.iteritems():
                plt.scatter(value[:,0], value[:,1], alpha=0.5, label=key)

            if plot_background:
                plt.scatter(descriptors_background_samples[:,0], descriptors_background_samples[:,1], alpha=0.5, label="background")
            plt.legend()
            plt.show()
        
        if use_3d:
            for key, value in descriptors_known_objects_samples_xy.iteritems():
                plt.scatter(value[:,0], value[:,1], alpha=0.5, label=key)
            if plot_background:
                plt.scatter(descriptors_background_samples_xy[:,0], descriptors_background_samples_xy[:,1], alpha=0.5, label="background")
            plt.legend()
            plt.show()

            for key, value in descriptors_known_objects_samples_yz.iteritems():
                plt.scatter(value[:,0], value[:,1], alpha=0.5, label=key)
            if plot_background:
                plt.scatter(descriptors_background_samples_yz[:,0], descriptors_background_samples_yz[:,1], alpha=0.5, label="background")
            plt.legend()
            plt.show()

            for key, value in descriptors_known_objects_samples_xz.iteritems():
                plt.scatter(value[:,0], value[:,1], alpha=0.5, label=key)
            if plot_background:
                plt.scatter(descriptors_background_samples_xz[:,0], descriptors_background_samples_xz[:,1], alpha=0.5, label="background")
            plt.legend()
            plt.show()

        print "done"

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
    """
    This class contains plotting utilities. They are all
    encapsulated as static methods

    """

    def __init__(self):
        pass

    @staticmethod
    def make_cdf_plot(ax, data, num_bins, label=None, x_axis_scale_factor=1):
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
        x_axis /= x_axis_scale_factor
        plot = ax.plot(x_axis, cumhist, label=label)
        return plot

    @staticmethod
    def make_pixel_match_error_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df: pandas dataframe, i.e. generated from quantitative 
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        
        if masked:
            data_string = 'pixel_match_error_l2_masked'
        else:
            data_string = 'pixel_match_error_l2' 

        data = df[data_string]

        # rescales the pixel distance to be relative to the diagonal of the image
        x_axis_scale_factor = 800

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label, x_axis_scale_factor=x_axis_scale_factor)
        if masked:
            ax.set_xlabel('Pixel match error (masked), L2 (pixel distance)')
        else:
            ax.set_xlabel('Pixel match error (fraction of image), L2 (pixel distance)')
        ax.set_ylabel('Fraction of images')

        ax.set_xlim([0,0.2])
        return plot

    @staticmethod
    def make_across_object_best_match_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df: pandas dataframe, i.e. generated from quantitative 
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['norm_diff_descriptor_best_match']

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Best descriptor match, L2 norm')
        ax.set_ylabel('Fraction of pixel samples from images')
        return plot

    @staticmethod
    def make_descriptor_accuracy_plot(ax, df, label=None, num_bins=100, masked=False):
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

        if masked:
            data_string = 'norm_diff_pred_3d_masked'
        else:
            data_string = 'norm_diff_pred_3d' 


        data = df[data_string]
        data = data.dropna()
        data *= 100 # convert to cm

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        if masked:
            ax.set_xlabel('3D match error (masked), L2 (cm)')
        else:
            ax.set_xlabel('3D match error, L2 (cm)')
        ax.set_ylabel('Fraction of images')
        #ax.set_title("3D Norm Diff Best Match")
        return plot

    @staticmethod
    def make_norm_diff_ground_truth_plot(ax, df, label=None, num_bins=100, masked=False):
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
    def make_fraction_false_positives_plot(ax, df, label=None, num_bins=100, masked=False):
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

        if masked:
            data_string = 'fraction_pixels_closer_than_ground_truth_masked'
        else:
            data_string = 'fraction_pixels_closer_than_ground_truth' 

        data = df[data_string]
        
        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        
        if masked:
            ax.set_xlabel('Fraction false positives (masked)')
        else:
            ax.set_xlabel('Fraction false positives')    

        ax.set_ylabel('Fraction of images')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return plot

    @staticmethod
    def make_average_l2_false_positives_plot(ax, df, label=None, num_bins=100, masked=False):
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

        if masked:
            data_string = 'average_l2_distance_for_false_positives_masked'
        else:
            data_string = 'average_l2_distance_for_false_positives'

        data = df[data_string]
        
        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        if masked:
            ax.set_xlabel('Average l2 pixel distance for false positives (masked)')
        else:
            ax.set_xlabel('Average l2 pixel distance for false positives')
        ax.set_ylabel('Fraction of images')
        # ax.set_xlim([0,200])
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
    def run_on_single_dataframe(path_to_df_csv, label=None, output_dir=None, save=True, previous_fig_axes=None, dataframe=None):
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

        :param dataframe: The pandas dataframe, object
        :type dataframe:
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

        if dataframe is None:
            path_to_csv = utils.convert_data_relative_path_to_absolute_path(path_to_df_csv,
                assert_path_exists=True)
            df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)
            if output_dir is None:
                output_dir = os.path.dirname(path_to_csv)
        else:
            df = dataframe
            if save and (output_dir is None):
                raise ValueError("You must pass in an output directory")


        if 'is_valid_masked' not in df:
            use_masked_plots = False
        else:
            use_masked_plots = True
        

        if previous_fig_axes==None:
            N = 5
            if use_masked_plots:
                fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(15,N*5))
            else:
                fig, axes = plt.subplots(N, figsize=(10,N*5))
        else:
            [fig, axes] = previous_fig_axes
        
        
        def get_ax(axes, index):
            if use_masked_plots:
                return axes[index,0]
            else:
                return axes[index]

        # pixel match error
        ax = get_ax(axes, 0)
        plot = DCEP.make_pixel_match_error_plot(ax, df, label=label)
        if use_masked_plots:
            plot = DCEP.make_pixel_match_error_plot(axes[0,1], df, label=label, masked=True)
        ax.legend()
       
        # 3D match error
        ax = get_ax(axes, 1)
        plot = DCEP.make_descriptor_accuracy_plot(ax, df, label=label)
        if use_masked_plots:
            plot = DCEP.make_descriptor_accuracy_plot(axes[1,1], df, label=label, masked=True)            


        # if save:
        #     fig_file = os.path.join(output_dir, "norm_diff_pred_3d.png")
        #     fig.savefig(fig_file)

        aac = DCEP.compute_area_above_curve(df, 'norm_diff_pred_3d')
        d = dict()
        d['norm_diff_3d_area_above_curve'] = float(aac)

        # norm difference of the ground truth match (should be 0)
        ax = get_ax(axes,2)
        plot = DCEP.make_norm_diff_ground_truth_plot(ax, df, label=label)

        # fraction false positives
        ax = get_ax(axes,3)
        plot = DCEP.make_fraction_false_positives_plot(ax, df, label=label)
        if use_masked_plots:
            plot = DCEP.make_fraction_false_positives_plot(axes[3,1], df, label=label, masked=True)

        # average l2 false positives
        ax = get_ax(axes, 4)
        plot = DCEP.make_average_l2_false_positives_plot(ax, df, label=label)
        if use_masked_plots:
            plot = DCEP.make_average_l2_false_positives_plot(axes[4,1], df, label=label, masked=True)

        if save:
            yaml_file = os.path.join(output_dir, 'stats.yaml')
            utils.saveToYaml(d, yaml_file)
        return [fig, axes]

    @staticmethod
    def run_on_single_dataframe_across_objects(path_to_df_csv, label=None, output_dir=None, save=True, previous_fig_axes=None):
        """
        This method is intended to be called from an ipython notebook for plotting.

        See run_on_single_dataframe() for documentation.

        The only difference is that for this one, we only have across object data. 
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        path_to_csv = utils.convert_data_relative_path_to_absolute_path(path_to_df_csv,
            assert_path_exists=True)

        if output_dir is None:
            output_dir = os.path.dirname(path_to_csv)

        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)

        if previous_fig_axes==None:
            N = 1
            fig, ax = plt.subplots(N, figsize=(10,N*5))
        else:
            [fig, ax] = previous_fig_axes
        
        
        # pixel match error
        plot = DCEP.make_across_object_best_match_plot(ax, df, label=label)
        ax.legend()
       
        if save:
            fig_file = os.path.join(output_dir, "across_objects.png")
            fig.savefig(fig_file)
        
        return [fig, ax]
        


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
