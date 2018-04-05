#!/usr/bin/python


import os
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd


from dense_correspondence_manipulation.utils.constants import *
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

import dense_correspondence.evaluation.plotting as dc_plotting

from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image


class DenseCorrespondenceEvaluation(object):


    def __init__(self, config):
        self._config = config

    def evaluate_network(self, network_data_dict):
        """

        :param network_data_dict: Dict with fields
            - path_to_network
            - parameter_file
            - descriptor_dimensionality
        :return:
        """
        pass

    def load_network_from_config(self, name):
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))


        network_config = self._config["networks"][name]
        return DenseCorrespondenceNetwork.from_config(network_config)

    @staticmethod
    def evaluate_network(nn, test_dataset):
        """

        :param nn: A neural network
        :param test_dataset: DenseCorrespondenceDataset
            the dataset to draw samples from
        :return:
        """
        pass

    @staticmethod
    def get_image_data(dataset, scene_name, img_idx):
        """
        Gets RGBD image, pose, mask

        :param dataset: dataset that has image
        :type dataset: DenseCorrespondenceDataset
        :param scene_name: name of scene
        :type scene_name: str
        :param img_idx: index of the image
        :type img_idx: int
        :return: (rgb, depth, mask, pose)
        :rtype: (PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, numpy.ndarray)
        """

        img_idx = utils.getPaddedString(img_idx)
        rgb = dataset.get_rgb_image_from_scene_name_and_idx(scene_name, img_idx)
        depth = dataset.get_depth_image_from_scene_name_and_idx(scene_name, img_idx)
        mask = dataset.get_mask_image_from_scene_name_and_idx(scene_name, img_idx)
        pose = dataset.get_pose_from_scene_name_and_idx(scene_name, img_idx)

        return rgb, depth, mask, pose

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
                                                params=None, camera_intrinsics_matrix=None,
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

        rgb_a, depth_a, mask_a, pose_a = DenseCorrespondenceEvaluation.get_image_data(dataset,
                                                                                      scene_name,
                                                                                      img_a_idx)

        rgb_b, depth_b, mask_b, pose_b = DenseCorrespondenceEvaluation.get_image_data(dataset,
                                                                                      scene_name,
                                                                                      img_b_idx)

        # compute dense descriptors
        res_a = dcn.forward_on_img(rgb_a)
        res_b = dcn.forward_on_img(rgb_b)

        if camera_intrinsics_matrix is None:
            camera_intrinsics = dataset.get_camera_intrinsics(scene_name)
            camera_intrinsics_matrix = camera_intrinsics.K


        # find correspondences
        # what type does img_a_depth need to be?
        (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(depth_a, pose_a, depth_b, pose_b,
                                                               device='CPU', img_a_mask=mask_a)




        # create pandas dataframe
        data_frame = pd.DataFrame()

        num_matches = uv_a_vec[0].size()[0]
        match_list = range(0, num_matches)
        if debug:
            match_list = [50, 1000]

        logging_rate = 100

        image_height, image_width = dcn.image_shape
        def clip_pixel_to_image_size(uv):
            u = min(int(round(uv[0])), image_width - 1)
            v = min(int(round(uv[1])), image_height - 1)
            return [u,v]


        for i in match_list:
            uv_a = [uv_a_vec[0][i], uv_a_vec[1][i]]
            uv_b_raw = [uv_b_vec[0][i], uv_b_vec[1][i]]
            uv_b = clip_pixel_to_image_size(uv_b_raw)

            d, series_data = DenseCorrespondenceEvaluation.compute_descriptor_match_statistics(depth_a,
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

            series_data['scene_name'] = scene_name
            series_data['img_a_idx'] = int(img_a_idx)
            series_data['img_b_idx'] = int(img_b_idx)

            series = pd.Series(series_data)

            # very inefficient but ok for now
            data_frame = data_frame.append(series, ignore_index=True)

            if i % logging_rate == 0:
                print "computing statistics for match %d of %d" %(i, num_matches)
            # if i > 10:
            #     break

        return data_frame

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
                                                       res_b)

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
        df_dict = dict()
        df_dict['norm_diff_descriptor'] = best_match_diff
        df_dict['is_valid'] = is_valid
        df_dict['norm_diff_ground_truth_3d'] = norm_diff_ground_truth_3d
        df_dict['norm_diff_pred_3d'] = norm_diff_pred_3d

        return d, df_dict

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

        rgb_a, depth_a, mask_a, pose_a = DenseCorrespondenceEvaluation.get_image_data(dataset,
                                                                                      scene_name,
                                                                                      img_a_idx)

        rgb_b, depth_b, mask_b, pose_b = DenseCorrespondenceEvaluation.get_image_data(dataset,
                                                                                      scene_name,
                                                                                      img_b_idx)

        # compute dense descriptors
        res_a = dcn.forward_on_img(rgb_a)
        res_b = dcn.forward_on_img(rgb_b)

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
        plt.show()



        # show colormap if possible (i.e. if descriptor dimension is 1 or 3)
        if dcn.descriptor_dimension in [1,3]:
            DenseCorrespondenceEvaluation.plot_descriptor_colormaps(res_a, res_b)




    @staticmethod
    def evaluate_network_qualitative(dcn, num_image_pairs=5, randomize=False, dataset=None):

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
            scene_name = '13_drill_long_downsampled'
            img_pairs = []
            img_pairs.append([0,737])
            img_pairs.append([409, 1585])
            img_pairs.append([2139, 1041])
            img_pairs.append([235, 1704])

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
            scene_name = '06_drill_long_downsampled'
            img_pairs = []
            img_pairs.append([0, 617])
            img_pairs.append([270, 786])
            img_pairs.append([1001, 2489])
            img_pairs.append([1536, 1917])


        for img_pair in img_pairs:
            print "Image pair (%d, %d)" %(img_pair[0], img_pair[1])
            DenseCorrespondenceEvaluation.single_image_pair_qualitative_analysis(dcn,
                                                                                 dataset,
                                                                                 scene_name,
                                                                                 img_pair[0],
                                                                                 img_pair[1])

    @staticmethod
    def make_default():
        """
        Makes a DenseCorrespondenceEvaluation object using the default config
        :return:
        :rtype: DenseCorrespondenceEvaluation
        """
        config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'evaluation.yaml')
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



def run():
    pass

def main(config):
    eval = DenseCorrespondenceEvaluation(config)
    dcn = eval.load_network_from_config("10_scenes_drill")
    test_dataset = SpartanDataset(mode="test")

    DenseCorrespondenceEvaluation.test(dcn, test_dataset)

def test():
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'evaluation.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    default_config = utils.get_defaults_config()
    utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

    main(config)

if __name__ == "__main__":
    test()