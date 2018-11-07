import sys
import os
import cv2
import numpy as np

# pdc
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.image_utils as image_utils
import dense_correspondence_manipulation.utils.visualization as vis_utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence_manipulation.utils.constants as constants

DC_SOURCE_DIR = utils.getDenseCorrespondenceSourceDir()
CONFIG_FILE = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence',
                           'keypoints', 'shoe_keypoints.yaml')

EVAL_CONFIG_FILENAME = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence', 'evaluation', 'lucas_evaluation.yaml')


class KeypointDetection(object):

    def __init__(self, config, eval_config):
        self._config = config
        self._eval_config = eval_config
        self._dce = DenseCorrespondenceEvaluation(eval_config)
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
        self._source_window_name = 'source'
        cv2.namedWindow(self._source_window_name)
        for network_name in self._dcn_dict:
            cv2.namedWindow(network_name)

    def _visualize_reference_image(self):
        """
        Visualize the reference image, with reticles at the keypoints
        :return:
        :rtype: cv2 image
        """

        img = None

        for keypoint, data in self._config["keypoints"].iteritems():
            scene_name = data['scene_name']
            u = data['u']
            v = data['v']
            image_idx = data['image_idx']
            color = data['color']

            if img is None:
                # load the image
                rgb_pil, _, _, _ = self._dataset.get_rgbd_mask_pose(scene_name, image_idx)
                img = image_utils.pil_image_to_cv2(rgb_pil)

            vis_utils.draw_reticle(img, u, v, color)

        print "scene_name", scene_name
        print "image_idx", image_idx

        return img


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
        :return: dict of results, keys are the names of the keypoints
        :rtype:
        """
        keypoint_detections = dict()

        for keypoint, keypoint_data in self._config['keypoints'].iteritems():
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
        rgb_tensor = self._dataset.rgb_image_to_tensor(img_pil)
        for network_name, dcn in self._dcn_dict.iteritems():
            self._res[network_name] = dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()

    def _visualize_keypoints(self, img, keypoint_detections):
        """
        Overlay the  the keypoint detections onto the image
        :param img:
        :type img: cv2 image
        :param keypoint_detections:
        :type keypoint_detections:
        :return: cv2 image with keypoint detections
        :rtype:
        """

        img_w_keypoints = np.copy(img)

        print "type(img_w_keypoints)", type(img_w_keypoints)
        print "img_w_keypoints.shape", img_w_keypoints.shape

        for keypoint, data in keypoint_detections.iteritems():
            color = self._config["keypoints"][keypoint]["color"]
            u, v = data['best_match_uv']
            vis_utils.draw_reticle(img_w_keypoints, u, v, color)

        return img_w_keypoints

    def _step(self):
        img_pil, scene_name, image_idx = self._get_random_image()
        img = image_utils.pil_image_to_cv2(img_pil)
        self._res = dict()
        self._compute_descriptors(img_pil)


        for network_name in self._dcn_dict:
            # for each network, compute the descriptor image
            # detect keypoints, visualize the keypoints
            # note: there should only be one network, but it's easy to do it this way
            print "network_name:", network_name
            res = self._res[network_name]
            keypoint_list = self._detect_keypoints(res)
            img_w_keypoints = self._visualize_keypoints(img, keypoint_list)

            print "img_w_keypoints.shape", img_w_keypoints.shape
            cv2.imshow(network_name, img_w_keypoints)

    def run(self):
        self._construct_windows()
        reference_image = self._visualize_reference_image()
        cv2.imshow(self._source_window_name, reference_image)

        self._step()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                print "Getting new target image"
                self._step()

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


        eval_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'lucas_evaluation.yaml')

        eval_config = utils.getDictFromYamlFilename(eval_config_file)

        kp = KeypointDetection(config, eval_config)
        return kp



if __name__ == "__main__":
    config = utils.getDictFromYamlFilename(CONFIG_FILE)
    eval_config = utils.getDictFromYamlFilename(EVAL_CONFIG_FILENAME)
    keypoint_detection_vis = KeypointDetection(config, eval_config)
    print "starting keypoint vis"
    keypoint_detection_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()