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

sys.path.append(os.path.join(os.path.dirname(__file__), "../simple-pixel-correspondence-labeler"))
from annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2



COLOR_RED = np.array([0, 0, 255])

utils.set_default_cuda_visible_devices()
eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'lucas_evaluation.yaml')
EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)


class HeatmapVisualization(object):

    def __init__(self):
        self._dataset = SpartanDataset.make_default_caterpillar()
        self._dce = DenseCorrespondenceEvaluation(EVAL_CONFIG)
        self._dcn = self._dce.load_network_from_config("caterpillar_background_0.5_3")

        self._config = dict()
        self._config["norm_diff_threshold"] = 0.25
        self._reticle_color = np.array([0, 0, 255])
        self._blend_weight_original_image = 0.3

    def get_random_image_pair(self, randomize_image=False):
        object_id = self._dataset.get_random_object_id()
        scene_name_a = self._dataset.get_random_single_object_scene_name(object_id)
        scene_name_b = self._dataset.get_different_scene_for_object(object_id, scene_name_a)

        if randomize_image:
            pass
        else:
            image_a_idx = 0
            image_b_idx = 0

        return scene_name_a, scene_name_b, image_a_idx, image_b_idx

    def _get_new_images(self):
        scene_name_1, scene_name_2, image_1_idx, image_2_idx = self.get_random_image_pair()

        img1_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx)
        img2_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name_2, image_2_idx)

        rgb_1_tensor = self._dataset.rgb_image_to_tensor(img1_pil)
        rgb_2_tensor = self._dataset.rgb_image_to_tensor(img2_pil)

        self.res_a = self._dcn.forward_single_image_tensor(rgb_1_tensor).data.cpu().numpy()
        self.res_b = self._dcn.forward_single_image_tensor(rgb_2_tensor).data.cpu().numpy()

        self.img1_descriptors = numpy_to_cv2(self.res_a)
        self.img2_descriptors = numpy_to_cv2(self.res_b)

        self.img1 = pil_image_to_cv2(img1_pil)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)/255.0
        self.img2 = pil_image_to_cv2(img2_pil)
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)/255.0

        cv2.imshow('image1', self.img1)
        cv2.imshow('image2', self.img2)

        self.find_best_match_from_image_1(None, 0, 0, None, None)
        self.find_best_match_from_image_2(None, 0, 0, None, None)



    def scale_norm_diffs_to_make_heatmap(self, norm_diffs):
        """
        Scales the norm diffs to make a heatmap. This will be scaled between 0 and 1.
        0 corresponds to a match, 1 to non-match

        :param norm_diffs: The norm diffs
        :type norm_diffs: numpy.array [H,W]
        :return:
        :rtype:
        """

        threshold = self._config["norm_diff_threshold"]
        heatmap = np.copy(norm_diffs)
        greater_than_threshold = np.where(norm_diffs > threshold)
        heatmap = heatmap / threshold * 0.5 # linearly scale [0, threshold] to [0, 0.5]
        heatmap[greater_than_threshold] = 1 # greater than threshold is set to 1
        heatmap = heatmap.astype(self.img1_gray.dtype)
        return heatmap


    def find_best_match_from_image_1(self, event,u,v,flags,param):
        """
        This should update the heatmap for image 2, self.img2_heatmap
        :return:
        :rtype:
        """

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("image1", img_1_with_reticle)

        best_match_uv, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match((u, v), self.res_a, self.res_b)

        print "best_match_diff", best_match_diff

        self.img2_heatmap = self.scale_norm_diffs_to_make_heatmap(norm_diffs)
        draw_reticle(self.img2_heatmap, best_match_uv[0], best_match_uv[1], self._reticle_color)

        alpha = self._blend_weight_original_image
        beta = 1 - self._blend_weight_original_image

        blended = cv2.addWeighted(self.img2_gray, alpha,
                                  self.img2_heatmap, beta, 0)

        img_2_with_reticle = np.copy(self.img2)
        draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], self._reticle_color)

        cv2.imshow("image2", img_2_with_reticle)

        cv2.imshow('image2_heatmap', blended)

    def find_best_match_from_image_2(self, event,u,v,flags,param):
        """
        This should update the heatmap for image 2, self.img2_heatmap
        :return:
        :rtype:
        """

        img_2_with_reticle = np.copy(self.img2)
        draw_reticle(img_2_with_reticle, u, v, self._reticle_color)
        cv2.imshow("image2", img_2_with_reticle)

        best_match_uv, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match((u, v), self.res_b, self.res_a)

        print "best_match_diff", best_match_diff

        self.img1_heatmap = self.scale_norm_diffs_to_make_heatmap(norm_diffs)
        draw_reticle(self.img1_heatmap, best_match_uv[0], best_match_uv[1], self._reticle_color)


        alpha = self._blend_weight_original_image
        beta = 1 - self._blend_weight_original_image
        blended = cv2.addWeighted(self.img1_gray, alpha,
                                  self.img1_heatmap, beta, 0)

        cv2.imshow('image1_heatmap', blended)

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, best_match_uv[0], best_match_uv[1], self._reticle_color)

        cv2.imshow("image1", img_1_with_reticle)


    def run(self):
        cv2.namedWindow('image1')
        cv2.setMouseCallback('image1', self.find_best_match_from_image_1)
        cv2.namedWindow('image1_heatmap')

        cv2.namedWindow('image2')
        cv2.setMouseCallback('image2', self.find_best_match_from_image_2)
        cv2.namedWindow('image2_heatmap')
        self._get_new_images()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                print "HEY"
                self._get_new_images()


if __name__ == "__main__":
    heatmap_vis = HeatmapVisualization()
    print "starting heatmap vis"
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()