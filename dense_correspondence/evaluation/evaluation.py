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

from torch.autograd import Variable



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
            'pixel_match_error']

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
                DCE.single_image_pair_quantitative_analysis(dcn, dataset, scene_name,
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
    def plot_descriptor_colormaps(res_a, res_b):
        """
        Plots the colormaps of descriptors for a pair of images
        :param res_a: descriptors for img_a
        :type res_a: numpy.ndarray
        :param res_b:
        :type res_b: numpy.ndarray
        :return: None
        :rtype: None
        """

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        res_a_norm = dc_plotting.normalize_descriptor(res_a)
        axes[0].imshow(res_a_norm)

        res_b_norm = dc_plotting.normalize_descriptor(res_b)
        axes[1].imshow(res_b_norm)

    @staticmethod
    def single_image_pair_quantitative_analysis(dcn, dataset, scene_name,
                                                img_a_idx, img_b_idx,
                                                params=None,
                                                camera_intrinsics_matrix=None,
                                                num_matches=100,
                                                debug=False):
        """


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
        :param params:
        :type params:
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
        # what type does img_a_depth need to be?
        (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(depth_a, pose_a, depth_b, pose_b,
                                                               device='CPU', img_a_mask=mask_a)


        if uv_a_vec is None:
            print "no matches found, returning"
            return None

        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []
        dataframe_list = []


        total_num_matches = len(uv_a_vec[0])
        match_list = random.sample(range(0, total_num_matches), num_matches)

        if debug:
            match_list = [50]

        logging_rate = 100

        image_height, image_width = dcn.image_shape
        def clip_pixel_to_image_size_and_round(uv):
            u = min(int(round(uv[0])), image_width - 1)
            v = min(int(round(uv[1])), image_height - 1)
            return [u,v]


        for i in match_list:
            uv_a = [uv_a_vec[0][i], uv_a_vec[1][i]]
            uv_b_raw = [uv_b_vec[0][i], uv_b_vec[1][i]]
            uv_b = clip_pixel_to_image_size_and_round(uv_b_raw)

            d, pd_template = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(depth_a,
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
            pd_template.set_value('img_a_idx',  int(img_a_idx))
            pd_template.set_value('img_b_idx', int(img_b_idx))


            dataframe_list.append(pd_template.dataframe)


            # if i % logging_rate == 0:
            #     print "computing statistics for match %d of %d" %(i, num_matches)
            # if i > 10:
            #     break

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

        :param uv_a:
        :type uv_a:
        :param uv_b:
        :type uv_b:
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
        :param res_a:
        :type res_a:
        :param res_b:
        :type res_b:
        :param params:
        :type params:
        :param debug: whether or not to print visualization
        :type debug:
        :return:
        :rtype:
        """

        DCE = DenseCorrespondenceEvaluation

        d = dict()
        # compute best match

        uv_b_pred, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b, debug=debug)

        # extract the ground truth descriptors
        des_a = res_a[uv_a[1], uv_a[0], :]
        des_b_ground_truth = res_b[uv_b[1], uv_b[0], :]
        norm_diff_descriptor_ground_truth = np.linalg.norm(des_a - des_b_ground_truth)


        # print "type(depth_a): ", type(depth_a)
        # # print "depth_a.shape(): ", depth_a.shape()
        # print "uv_a:", uv_a
        # print "uv_b: ", uv_b
        # print "uv_b_pred: ", uv_b_pred



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

        diff_pred_3d = uv_a_pos - uv_b_pred_pos

        if DCE.is_depth_valid(uv_b_depth):
            norm_diff_ground_truth_3d = np.linalg.norm(diff_ground_truth_3d)
        else:
            norm_diff_ground_truth_3d = np.nan

        if is_valid:
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



        # construct a dict with the return data, which will later be put into a pandas.DataFrame object
        d = dict()

        d['uv_a'] = uv_a
        d['uv_b'] = uv_b
        d['uv_b_pred'] = uv_b_pred
        d['norm_diff_descriptor'] = best_match_diff

        d['pose_a'] = pose_a
        d['pose_b'] = pose_b

        d['uv_a_depth'] = uv_a_depth
        d['uv_b_depth'] = uv_b_depth
        d['uv_b_pred_depth'] = uv_b_pred_depth
        d['uv_b_pred_depth_is_valid'] = uv_b_pred_depth_is_valid

        d['is_valid'] = is_valid

        d['uv_a_pos'] = uv_a_pos
        d['uv_b_pos'] = uv_b_pos
        d['uv_b_pre_pos'] = uv_b_pred_pos

        d['diff_ground_truth_3d'] = diff_ground_truth_3d
        d['norm_diff_ground_truth_3d'] = norm_diff_ground_truth_3d
        d['diff_pred_3d'] = diff_pred_3d
        d['norm_diff_pred_3d'] = norm_diff_pred_3d

        # dict for making a pandas.DataFrame later
        pd_template = DCNEvaluationPandaTemplate()
        pd_template.set_value('norm_diff_descriptor', best_match_diff)
        pd_template.set_value('is_valid', is_valid)

        pd_template.set_value('norm_diff_ground_truth_3d', norm_diff_ground_truth_3d)

        if is_valid:
            pd_template.set_value('norm_diff_pred_3d', norm_diff_pred_3d)
        else:
            pd_template.set_value('norm_diff_pred_3d', np.nan)

        pd_template.set_value('norm_diff_descriptor_ground_truth', norm_diff_ground_truth_3d)

        pixel_match_error = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=1)
        pd_template.set_value('pixel_match_error', pixel_match_error)

        return d, pd_template

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
    def single_image_pair_qualitative_analysis(dcn, dataset, scene_name,
                                               img_a_idx, img_b_idx,
                                               num_matches=10):
        """
        Computes qualtitative assessment of DCN performance for a pair of
        images

        :param dcn: dense correspondence network to use
        :param dataset: dataset to un the dataset
        :param num_matches: number of matches to generate
        :param scene_name: scene name to use
        :param img_a_idx: index of image_a in the dataset
        :param img_b_idx: index of image_b in the datset


        :type dcn: DenseCorrespondenceNetwork
        :type dataset: DenseCorrespondenceDataset
        :type num_matches: int
        :type scene_name: str
        :type img_a_idx: int
        :type img_b_idx: int
        :type num_matches: int

        :return: None
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
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).cpu().data.numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).cpu().data.numpy()


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
            DenseCorrespondenceEvaluation.plot_descriptor_colormaps(res_a, res_b)

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
    def evaluate_network_qualitative(dcn, num_image_pairs=5, randomize=False, dataset=None,
                                     scene_type="caterpillar"):

        if dataset is None:
            config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset',
                                       'spartan_dataset_masked.yaml')

            config = utils.getDictFromYamlFilename(config_file)

            dataset = SpartanDataset(mode="test", config=config)

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
            DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn,
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
            DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn,
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
                DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn,
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

        DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn, dataset,
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
    def make_cdf_plot(data, label=None, num_bins=30):
        """
        Plots the empirical CDF of the data
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
        plot = plt.plot(x_axis, cumhist, label=label)
        return plot

    @staticmethod
    def make_descriptor_accuracy_plot(df, label=None, num_bins=30):
        """
        Makes a plot of best match accuracy.
        Drops nans
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

        plot = DCEP.make_cdf_plot(data, label=label, num_bins=num_bins)
        plt.xlabel('error (cm)')
        plt.ylabel('fraction below threshold')
        plt.title("3D Norm Diff Best Match")
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
    def run_on_single_dataframe(path_to_df_csv, label=None, output_dir=None, save=True, previous_plot=None):
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

        
        if previous_plot==None:
            fig = plt.figure()
        else:
            fig = previous_plot
        
        # norm diff accuracy
        plot = DCEP.make_descriptor_accuracy_plot(df, label=label)
        plt.legend()
        if save:
            fig_file = os.path.join(output_dir, "norm_diff_pred_3d.png")
            fig.savefig(fig_file)

        aac = DCEP.compute_area_above_curve(df, 'norm_diff_pred_3d')
        d = dict()
        d['norm_diff_3d_area_above_curve'] = float(aac)

        yaml_file = os.path.join(output_dir, 'stats.yaml')
        utils.saveToYaml(d, yaml_file)
        return fig


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