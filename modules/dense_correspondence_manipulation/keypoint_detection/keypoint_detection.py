import sys
import os
import cv2
import numpy as np

# pdc
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.image_utils as image_utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence_manipulation.utils.constants as constants

class KeypointDetection(object):

    def __init__(self, config, network_dict):
        self._config = config
        self._network_dict = network_dict
        self._dce = DenseCorrespondenceEvaluation(network_dict)
        self._dataset = None
        self._initialize()

    def _initialize(self):
        """
        Process the config
        :return:
        :rtype:
        """

        # add color data for each keypoint we are detecting
        counter = 0
        for keypoint, data in self._config["keypoints"].iteritems():
            data['color'] = constants.LABEL_COLORS[counter]
            counter += 1

        self._load_networks()
        self._construct_windows()

    def _load_networks(self):
        """
        Loads the networks specified in the config
        :return:
        :rtype:
        """
        self._dcn_dict = dict()

        self._network_reticle_color = dict()

        network_name = self._config["network_name"]
        dcn = self._dce.load_network_from_config(network_name)
        dcn.eval()
        self._dcn_dict[network_name] = dcn
        self._network_reticle_color[network_name] = constants.LABEL_COLORS[0]
        self._dataset = dcn.load_training_dataset()

        self._window_names = dict()
        self._window_names[network_name] = network_name

    def _construct_windows(self):
        """
        Constructs the cv2 windows. One for source image,
        and for each of the windows
        :return:
        :rtype:
        """
        cv2.namedWindow('source')
        for network_name in self._network_dict:
            cv2.namedWindow(network_name)


    def _get_random_image(self, randomize_images=False):
        object_id = self._dataset.get_random_object_id()
        scene_name = self._dataset.get_random_single_object_scene_name(object_id)

        if randomize_images:
            image_idx = self._dataset.get_random_image_index(scene_name)
        else:
            image_idx = 0

        img_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name, image_idx)
        return img_pil, scene_name, image_idx


    def _detect_keypoints(self, res):
        """
        Detect the keypoints (specified in the config) from the descriptor
        image res
        :param res: array of dense descriptors res = [H,W,D]
        :type res: numpy array with shape [H,W,D]
        :return:
        :rtype:
        """
        keypoint_detections = dict()

        for keypoint, keypoint_data in self._config['keypoints']:
            d = dict()
            keypoint_detections[keypoint] = d
            descriptor = keypoint_data['descriptor']
            best_match_uv, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match_for_descriptor(descriptor, res)

            d['best_match_uv'] = best_match_uv
            d['best_match_diff'] = best_match_diff
            d['norm_diffs'] = norm_diffs

        return keypoint_detections

    def _compute_descriptors(self, img_pil):
        """
        Computes the descriptor images for each network
        :param img_pil:
        :type img_pil:
        :return:
        :rtype:
        """
        rgb_tensor = self._dataset.image_to_tensor(img_pil)
        for network_name, dcn in self._dcn_dict.iteritems():
            self._res[network_name] = dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()

    def _visualize_keypoints(self, img, keypoint_detections, network_name):
        """
        Visualize the results of the network in the given window
        :param img:
        :type img: cv2 image
        :param network_name:
        :type network_name:
        :param keypoint_detections:
        :type keypoint_detections:
        :return:
        :rtype:
        """

        img_w_keypoints = np.copy(img)

        for keypoint in keypoint_detections:
            color = self._config["keypoints"][keypoint]["color"]
            img_w_keypoints


        pass



    def _step(self):
        img_pil, scene_name, image_idx = self._get_random_image()
        img = image_utils.pil_image_to_cv2(img_pil)
        self._res = dict()

        # descriptor image
        for network_name in self._network_dict:
            res = self._res[network_name]
            keypoint_list = self._detect_keypoints(res)
            self._visualize_keypoints(None, keypoint_list, network_name)

    @staticmethod
    def make_default():
        """
        Make a default KeypointDetection object from the shoe_keypoints.yaml
        config
        :return: 
        :rtype: 
        """
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        config_filename = os.path.join(dc_source_dir, 'config',
                                       'dense_correspondence',
                                       'keypoint_detection',
                                       'shoe_keypoints.yaml')

        config = utils.getDictFromYamlFilename(config_filename)


        network_dict_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'lucas_evaluation.yaml')

        network_dict = utils.getDictFromYamlFilename(network_dict_file)

        kp = KeypointDetection(config, network_dict)
        return kp


