import sys
import os
import cv2
import numpy as np
import copy

import dense_correspondence_manipulation.utils.utils as utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType

import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
from dense_correspondence.evaluation.plotting import normalize_descriptor
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

sys.path.append(os.path.join(os.path.dirname(__file__), "../simple_pixel_correspondence_labeler"))
from annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2, label_colors



COLOR_RED = np.array([0, 0, 255])
COLOR_GREEN = np.array([0,255,0])

utils.set_default_cuda_visible_devices()
eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'lucas_evaluation.yaml')
EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)


class HeatmapVisualization(object):

    def __init__(self, config):
        self._config = config
        self._dce = DenseCorrespondenceEvaluation(EVAL_CONFIG)
        self._load_networks()
        self._reticle_color = COLOR_GREEN
        # self.load_specific_dataset() # uncomment if you want to load a specific dataset

    def _load_networks(self):
        # we will use the dataset for the first network in the series
        self._dcn_dict = dict()

        self._dataset = None
        self._network_reticle_color = dict()

        for idx, network_name in enumerate(self._config["networks"]):
            dcn = self._dce.load_network_from_config(network_name)
            dcn.eval()
            self._dcn_dict[network_name] = dcn
            # self._network_reticle_color[network_name] = label_colors[idx]

            if len(self._config["networks"]) == 1:
                self._network_reticle_color[network_name] = COLOR_RED
            else:
                self._network_reticle_color[network_name] = label_colors[idx]

            if self._dataset is None:
                self._dataset = dcn.load_training_dataset()

    def load_specific_dataset(self):
        dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                            'dataset', 'composite', 'hats_3_demo_composite.yaml')

        dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config',
                                               'dense_correspondence',
                                               'dataset', 'composite', '4_shoes_all.yaml')

        dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)
        self._dataset = SpartanDataset(config=dataset_config)

    def get_random_image_pair(self):
        """
        Gets a pair of random images for different scenes of the same object
        """
        object_id = self._dataset.get_random_object_id()
        # scene_name_a = "2018-04-10-16-02-59"
        # scene_name_b = scene_name_a

        scene_name_a = self._dataset.get_random_single_object_scene_name(object_id)
        scene_name_b = self._dataset.get_different_scene_for_object(object_id, scene_name_a)

        if self._config["randomize_images"]:
            image_a_idx = self._dataset.get_random_image_index(scene_name_a)
            image_b_idx = self._dataset.get_random_image_index(scene_name_b)
        else:
            image_a_idx = 0
            image_b_idx = 0


        # debug
        # image_a_idx = 0
        # image_b_idx = self._dataset.get_random_image_index(scene_name_b)

        

        # image_b_idx = self._dataset.get_random_image_index(scene_name_b)
        return scene_name_a, scene_name_b, image_a_idx, image_b_idx

    def get_random_image_pair_across_object(self):
        """
        Gets cross object image pairs
        :param randomize:
        :type randomize:
        :return:
        :rtype:
        """

        object_id_a, object_id_b = self._dataset.get_two_different_object_ids()
        # object_id_a = "shoe_red_nike.yaml"
        # object_id_b = "shoe_gray_nike"
        # object_id_b = "shoe_green_nike"
        scene_name_a = self._dataset.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self._dataset.get_random_single_object_scene_name(object_id_b)

        if self._config["randomize_images"]:
            image_a_idx = self._dataset.get_random_image_index(scene_name_a)
            image_b_idx = self._dataset.get_random_image_index(scene_name_b)
        else:
            image_a_idx = 0
            image_b_idx = 0

        return scene_name_a, scene_name_b, image_a_idx, image_b_idx

    def get_random_image_pair_multi_object_scenes(self):
        """
        Gets cross object image pairs
        :param randomize:
        :type randomize:
        :return:
        :rtype:
        """

        scene_name_a = self._dataset.get_random_multi_object_scene_name()
        scene_name_b = self._dataset.get_random_multi_object_scene_name()

        if self._config["randomize_images"]:
            image_a_idx = self._dataset.get_random_image_index(scene_name_a)
            image_b_idx = self._dataset.get_random_image_index(scene_name_b)
        else:
            image_a_idx = 0
            image_b_idx = 0

        return scene_name_a, scene_name_b, image_a_idx, image_b_idx

    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """

        if random.random() < 0.5:
            self._dataset.set_train_mode()
        else:
            self._dataset.set_test_mode()

        if self._config["same_object"]:
            scene_name_1, scene_name_2, image_1_idx, image_2_idx = self.get_random_image_pair()
        elif self._config["different_objects"]:
            scene_name_1, scene_name_2, image_1_idx, image_2_idx = self.get_random_image_pair_across_object()
        elif self._config["multiple_object"]:
            scene_name_1, scene_name_2, image_1_idx, image_2_idx = self.get_random_image_pair_multi_object_scenes()
        else:
            raise ValueError("At least one of the image types must be set tot True")


        self.img1_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx)
        self.img2_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name_2, image_2_idx)

        self._compute_descriptors()

        # self.rgb_1_tensor = self._dataset.rgb_image_to_tensor(img1_pil)
        # self.rgb_2_tensor = self._dataset.rgb_image_to_tensor(img2_pil)


    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        self.img1 = pil_image_to_cv2(self.img1_pil)
        self.img2 = pil_image_to_cv2(self.img2_pil)
        self.rgb_1_tensor = self._dataset.rgb_image_to_tensor(self.img1_pil)
        self.rgb_2_tensor = self._dataset.rgb_image_to_tensor(self.img2_pil)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY) / 255.0

        cv2.imshow('source', self.img1)
        cv2.imshow('target', self.img2)

        self._res_a = dict()
        self._res_b = dict()
        for network_name, dcn in self._dcn_dict.iteritems():
            self._res_a[network_name] = dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
            self._res_b[network_name] = dcn.forward_single_image_tensor(self.rgb_2_tensor).data.cpu().numpy()


        self.find_best_match(None, 0, 0, None, None)

    def scale_norm_diffs_to_make_heatmap(self, norm_diffs, threshold):
        """
        Scales the norm diffs to make a heatmap. This will be scaled between 0 and 1.
        0 corresponds to a match, 1 to non-match

        :param norm_diffs: The norm diffs
        :type norm_diffs: numpy.array [H,W]
        :return:
        :rtype:
        """


        heatmap = np.copy(norm_diffs)
        greater_than_threshold = np.where(norm_diffs > threshold)
        heatmap = heatmap / threshold * self._config["heatmap_vis_upper_bound"] # linearly scale [0, threshold] to [0, 0.5]
        heatmap[greater_than_threshold] = 1 # greater than threshold is set to 1
        heatmap = heatmap.astype(self.img1_gray.dtype)
        return heatmap


    def find_best_match(self, event,u,v,flags,param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("source", img_1_with_reticle)

        alpha = self._config["blend_weight_original_image"]
        beta = 1 - alpha

        img_2_with_reticle = np.copy(self.img2)


        print "\n\n"

        self._res_uv = dict()

        # self._res_a_uv = dict()
        # self._res_b_uv = dict()

        for network_name in self._dcn_dict:
            res_a = self._res_a[network_name]
            res_b = self._res_b[network_name]
            best_match_uv, best_match_diff, norm_diffs = \
                DenseCorrespondenceNetwork.find_best_match((u, v), res_a, res_b)
            print "\n\n"
            print "network_name:", network_name
            self._res_uv[network_name] = dict()
            self._res_uv[network_name]['source'] = res_a[v, u, :].tolist()
            self._res_uv[network_name]['target'] = res_b[v, u, :].tolist()

            # print "res_a[v, u, :]:", res_a[v, u, :]
            # print "res_b[v, u, :]:", res_b[v, u, :]

            print "%s best match diff: %.3f" %(network_name, best_match_diff)
            print "res_a", self._res_uv[network_name]['source']
            print "res_b", self._res_uv[network_name]['target']

            threshold = self._config["norm_diff_threshold"]
            if network_name in self._config["norm_diff_threshold_dict"]:
                threshold = self._config["norm_diff_threshold_dict"][network_name]

            heatmap = self.scale_norm_diffs_to_make_heatmap(norm_diffs, threshold)

            reticle_color = self._network_reticle_color[network_name]
            draw_reticle(heatmap, best_match_uv[0], best_match_uv[1], reticle_color)
            draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], reticle_color)
            blended = cv2.addWeighted(self.img2_gray, alpha, heatmap, beta, 0)
            cv2.imshow(network_name, blended)

        cv2.imshow("target", img_2_with_reticle)
        if event == cv2.EVENT_LBUTTONDOWN:
            utils.saveToYaml(self._res_uv, 'clicked_point.yaml')

    def run(self):
        self._get_new_images()
        cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        self._get_new_images()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._get_new_images()
            elif k == ord('s'):
                img1_pil = self.img1_pil
                img2_pil = self.img2_pil
                self.img1_pil = img2_pil
                self.img2_pil = img1_pil
                self._compute_descriptors()

if __name__ == "__main__":
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'heatmap_vis', 'heatmap.yaml')
    config = utils.getDictFromYamlFilename(config_file)

    heatmap_vis = HeatmapVisualization(config)
    print "starting heatmap vis"
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()