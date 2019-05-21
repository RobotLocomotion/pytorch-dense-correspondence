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
import dense_correspondence.network.dense_correspondence_network
from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image_torch


import dense_correspondence_manipulation.utils.visualization as vis_utils


from dense_correspondence_manipulation.simple_pixel_correspondence_labeler.annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2


COLOR_RED = np.array([0, 0, 255])
COLOR_GREEN = np.array([0,255,0])

#utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([1])
eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'evaluation.yaml')
EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)


LOAD_SPECIFIC_DATASET = False

class LogVisualization(object):


    def __init__(self, config):
        self._config = config
        self._dce = DenseCorrespondenceEvaluation(EVAL_CONFIG)
        self._load_networks()
        self._reticle_color = COLOR_GREEN
        self._paused = False
        if LOAD_SPECIFIC_DATASET:
            self.load_specific_dataset()
        if self._config["publish_to_ros"]:
            from ros_heatmap_visualizer import RosHeatmapVis
            self.ros_heatmap_vis = RosHeatmapVis()

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

    
    def _sample_new_reference_descriptor_pixels(self):
        num_samples = self._config["num_reference_descriptors"]
        self.img1_mask = torch.from_numpy(np.asarray(self._dataset.get_mask_image_from_scene_name_and_idx_and_cam(self.scene_name_1, self.image_1_idx, 0)))

        self.ref_pixels_uv = random_sample_from_masked_image_torch(self.img1_mask, num_samples) # tuple of (u's, v's)
        self.ref_pixels_flattened = self.ref_pixels_uv[1]*self.img1_mask.shape[1]+self.ref_pixels_uv[0]

        # DEBUG
        # import matplotlib.pyplot as plt
        # plt.imshow(np.asarray(self.img1_pil))
        # ref_pixels_uv_numpy = (ref_pixels_uv[0].numpy(), ref_pixels_uv[1].numpy())
        # plt.scatter(ref_pixels_uv_numpy[0], ref_pixels_uv_numpy[1])
        # plt.show()
        # sys.exit(0)

        # ref_pixels_flattened = ref_pixels_flattened.cuda()
        
        # # descriptor_image_reference starts out as H, W, D
        # D = descriptor_image_reference.shape[2]
        # WxH = descriptor_image_reference.shape[0]*descriptor_image_reference.shape[1]
        
        # # now switch it to D, H, W
        # descriptor_image_reference = descriptor_image_reference.permute(2, 0, 1)

        # # now view as D, H*W
        # descriptor_image_reference = descriptor_image_reference.contiguous().view(D, WxH)

        # # now switch back to H*W, D
        # descriptor_image_reference = descriptor_image_reference.permute(1,0)
        
        # # self.ref_descriptor_vec is Nref, D 
        # self.ref_descriptor_vec = torch.index_select(descriptor_image_reference, 0, ref_pixels_flattened)
        # self.ref_descriptor_vec.requires_grad_()

    def _get_new_reference(self):
        self.object_id = self._dataset.get_random_object_id()
        
        self.scene_name_1 = self._dataset.get_random_single_object_scene_name(self.object_id)
        
        if self._config["randomize_images"]:
            self.image_1_idx = self._dataset.get_random_image_index(self.scene_name_1)
        else:
            self.image_1_idx = 0

        self.img1_pil = self._dataset.get_rgb_image_from_scene_name_and_idx_and_cam(self.scene_name_1, self.image_1_idx, 0)

        if self._config["use_descriptor_tracks"]:
            self._sample_new_reference_descriptor_pixels()


    def _get_new_target_scene(self):
        self.scene_name_2  = self._dataset.get_different_scene_for_object(self.object_id, self.scene_name_1)
        
        #pose_data = self._dataset.get_pose_data(self.scene_name_2)
        #self.scene_2_indices =  sorted(pose_data.keys())

        scene_directory = self._dataset.get_full_path_for_scene(self.scene_name_2)
        state_info_filename = os.path.join(scene_directory, "states.yaml")
        state_info_dict = utils.getDictFromYamlFilename(state_info_filename)
        self.scene_2_indices = sorted(state_info_dict.keys()) # list of integers

        self.index_index = 0
        self.image_2_idx = self.scene_2_indices[self.index_index]


    def _get_next_target_image(self, increment):
        self.index_index += increment
        self.image_2_idx = self.scene_2_indices[self.index_index]

    def _get_new_images(self, increment):

        self._get_next_target_image(increment)
        image_2_idxs = [self.image_2_idx] * self._config["num_camera_target"]

        self.img2_pils = []
        self.img2_depth_np = []
        self.img2_poses = []
        self.img2_Ks = []

        for camera_num, image_2_idx in enumerate(image_2_idxs):
            self.img2_pils.append(self._dataset.get_rgb_image_from_scene_name_and_idx_and_cam(self.scene_name_2, image_2_idx, camera_num))
            if self._config["publish_to_ros"]:
                self.img2_depth_np.append(np.asarray(self._dataset.get_depth_image_from_scene_name_and_idx(self.scene_name_2, image_2_idx)))
                scene_pose_data = self._dataset.get_pose_data(self.scene_name_2)
                pose_data = scene_pose_data[image_2_idx]['camera_to_world']
                self.img2_poses.append(pose_data)
                self.img2_Ks.append(self._dataset.get_camera_intrinsics(self.scene_name_2).get_camera_matrix())

        self._scene_name_1 = self.scene_name_1
        self._scene_name_2 = self.scene_name_2
        self._image_1_idx = self.image_1_idx
        self._image_2_idxs = image_2_idxs

        self._compute_descriptors()


    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        self.img1 = pil_image_to_cv2(self.img1_pil)
        self.rgb_1_tensor = self._dataset.rgb_image_to_tensor(self.img1_pil)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0

        self.img2s = []
        self.img2s_gray = []
        self.rgb_2_tensors = []
        for img2_pil in self.img2_pils:
            self.img2s.append(pil_image_to_cv2(img2_pil))
            self.rgb_2_tensors.append(self._dataset.rgb_image_to_tensor(img2_pil))
            self.img2s_gray = cv2.cvtColor(self.img2s[-1], cv2.COLOR_RGB2GRAY) / 255.0

        cv2.imshow('source', self.img1)
        for i, v in enumerate(self.img2s):
            cv2.imshow('target'+str(i), v)

        self._res_a = dict()
        self._res_b = dict()
        for network_name, dcn in self._dcn_dict.iteritems():
            self._res_a[network_name] = dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
            self._res_b[network_name] = []
            for rgb_2_tensor in self.rgb_2_tensors:
                self._res_b[network_name].append(dcn.forward_single_image_tensor(rgb_2_tensor).data.cpu().numpy())

        if self._config["use_heatmap"]:
            self.find_best_match(None, self.last_u, self.last_v, None, None)
        elif self._config["use_descriptor_tracks"]:
            self.detect_reference_descriptors()

    def send_img_to_ros(self, topic_name, rgb, depth, pose, K):
        self.ros_heatmap_vis.update_rgb(topic_name, rgb, depth, pose, K)


    def get_color(self, index):
        r = (index * 55) % 255
        g = (150 + index * 35) % 255
        b = (255 - (index*85)) % 255
        return np.array([r, g, b])

    def detect_reference_descriptors(self):
        img_1_with_reticles = np.copy(self.img1)
        print self.ref_pixels_uv
        us = self.ref_pixels_uv[0]
        vs = self.ref_pixels_uv[1]
        i = 0
        for u, v in zip(us,vs):
            print u, v
            draw_reticle(img_1_with_reticles, u, v, self.get_color(i))
            i += 1 
        cv2.imshow("source", img_1_with_reticles)

        alpha = self._config["blend_weight_original_image"]
        beta = 1 - alpha

        img_2s_with_reticle = []
        for img2 in self.img2s:
            img_2s_with_reticle.append(np.copy(img2))

        for network_name in self._dcn_dict:
            res_a = self._res_a[network_name]
            dense_correspondence.network.dense_correspondence_network.COMPUTE_BEST_MATCH_WITH = self._dcn_dict[network_name].config["compute_best_match_with"]

            for image_num, res_b in enumerate(self._res_b[network_name]):


                i = 0
                for u, v in zip(us,vs):
                
                    best_match_uv, best_match_diff, norm_diffs = \
                        DenseCorrespondenceNetwork.find_best_match((u, v), res_a, res_b)


                    threshold = self._config["norm_diff_threshold"]

                    heatmap_color = vis_utils.compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, self._config['kernel_variance'])

                    
                    reticle_color = self.get_color(i)
                    draw_reticle(heatmap_color, best_match_uv[0], best_match_uv[1], reticle_color)
                    draw_reticle(img_2s_with_reticle[image_num], best_match_uv[0], best_match_uv[1], reticle_color)

                    blended = cv2.addWeighted(self.img2s[image_num], alpha, heatmap_color, beta, 0)
                    
                    window_name = network_name+"-target"+str(image_num)+"-d"+str(i)


                    
                    if self._config["display_all_separate_windows"]:
                        
                        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow(window_name, 200, 200)
                        cv2.imshow(window_name, blended)
                    else:
                        if i == 0:
                            blendeds = [blended]
                        else:
                            blendeds.append(blended)

                    
                    if self._config["publish_to_ros"]:
                        self.send_img_to_ros(window_name, blended, self.img2_depth_np[image_num], self.img2_poses[image_num], self.img2_Ks[image_num])

                    i += 1

                if not self._config["display_all_separate_windows"]:
                    
                    width = int(np.floor(np.sqrt(len(blendeds))))
                    height = len(blendeds)/width
                    print "width", width
                    print "height", height
                    
                    index = 0

                    for i in range(height):
                        for j in range(width):
                            
                            new = blendeds[index]
                            index += 1
                            
                            if j == 0:
                                hstack = new
                            else:
                                hstack = np.hstack((hstack, new))

                            print hstack.shape

                        if i == 0:
                            vstack = hstack
                        else:
                            vstack = np.vstack((vstack, hstack))

                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 640, 480)
                    cv2.imshow(window_name, vstack)

        for i, v in enumerate(img_2s_with_reticle):
            cv2.imshow("target"+str(i), v)

    def find_best_match(self, event,u,v,flags,param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """
        self.last_u = u
        self.last_v = v

        if self._paused:
            return

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("source", img_1_with_reticle)

        alpha = self._config["blend_weight_original_image"]
        beta = 1 - alpha

        img_2s_with_reticle = []
        for img2 in self.img2s:
            img_2s_with_reticle.append(np.copy(img2))


        #print "\n\n"

        self._res_uv = dict()

        # self._res_a_uv = dict()
        # self._res_b_uv = dict()

        for network_name in self._dcn_dict:
            res_a = self._res_a[network_name]
            dense_correspondence.network.dense_correspondence_network.COMPUTE_BEST_MATCH_WITH = self._dcn_dict[network_name].config["compute_best_match_with"]
            
            norm_diffs_list = []

            for image_num, res_b in enumerate(self._res_b[network_name]):
                
                best_match_uv, best_match_diff, norm_diffs = \
                    DenseCorrespondenceNetwork.find_best_match((u, v), res_a, res_b)

                norm_diffs_list.append(norm_diffs)

                self._res_uv[network_name] = dict()
                self._res_uv[network_name]['source'] = res_a[v, u, :].tolist()
                self._res_uv[network_name]['target'+str(image_num)] = res_b[v, u, :].tolist()


                threshold = self._config["norm_diff_threshold"]

                heatmap_color = vis_utils.compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, self._config['kernel_variance'])

                reticle_color = self._network_reticle_color[network_name]

                draw_reticle(heatmap_color, best_match_uv[0], best_match_uv[1], reticle_color)
                draw_reticle(img_2s_with_reticle[image_num], best_match_uv[0], best_match_uv[1], reticle_color)
                blended = cv2.addWeighted(self.img2s[image_num], alpha, heatmap_color, beta, 0)
                window_name = network_name+"-target"+str(image_num)
                cv2.imshow(window_name, blended)
                
                if self._config["publish_to_ros"]:
                    self.send_img_to_ros(window_name, blended, self.img2_depth_np[image_num], self.img2_poses[image_num], self.img2_Ks[image_num])

            # if self._config["multi_view"]:
            #     xyz = DenseCorrespondenceNetwork.find_multi_view_best_match(norm_diffs_list, self.img2_depth_np, self.img2_poses, self.img2_Ks)


        for i, v in enumerate(img_2s_with_reticle):
            cv2.imshow("target"+str(i), v)

        if event == cv2.EVENT_LBUTTONDOWN:
            utils.saveToYaml(self._res_uv, 'clicked_point.yaml')

    def run(self):
        self.last_u = 0
        self.last_v = 0
        self._get_new_reference()
        self._get_new_target_scene()

        self._get_new_images(increment=0)
        if self._config["use_heatmap"]:
            cv2.setMouseCallback('source', self.find_best_match)

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._get_new_images(increment=1)
            elif k == ord('b'):
                self._get_new_images(increment=-1)
            elif k == ord('r'):
                self._get_new_reference()
                self._get_new_images(increment=0)
            elif k == ord('t'):
                self._get_new_target_scene()
                self._get_new_images(increment=0)
            elif k == ord('p'):
                if self._paused:
                    print "un pausing"
                    self._paused = False
                else:
                    print "pausing"
                    self._paused = True


if __name__ == "__main__":
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'log_vis', 'log_vis.yaml')
    config = utils.getDictFromYamlFilename(config_file)
    
    if config["use_heatmap"] and config["use_descriptor_tracks"]:
        print "can't do both!"
        sys.exit(0)

    log_vis = LogVisualization(config)

    print "starting log vis"
    log_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
