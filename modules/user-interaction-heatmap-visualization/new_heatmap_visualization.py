import os
import cv2
import numpy as np
from random import randrange

# pdc
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

GPU_LIST = [0]
set_cuda_visible_devices(GPU_LIST)

# torch
import torch

# key_dynam
from key_dynam.utils.utils import load_yaml
import dense_correspondence_manipulation.utils.visualization as vis_utils

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir
import dense_correspondence.loss_functions.utils as loss_utils
import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence.network import predict

def load_dataset(episode_config, pdc_config):
    episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/logs_proto")
    episode_list_config = pdc_utils.getDictFromYamlFilename(episode_config)
    multi_episode_dict = SpartanEpisodeReader.load_dataset(episode_list_config, episodes_root)
    dataset = DynamicDrakeSimDataset(pdc_config, multi_episode_dict, phase='valid')
    return dataset


def load_model(model_file):
    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()
    return model


class HeatmapVisualization(object):
    def __init__(self, config_file, episode_file, model_file):
        pdc_config = load_yaml(config_file)
        self._dataset = load_dataset(episode_file, pdc_config)
        self._model = load_model(model_file)

        self._paused = False
        self._sigma = pdc_config['loss_function']['heatmap']['sigma_fraction']
        self._heatmap_type = pdc_config['loss_function']['heatmap']['heatmap_type']
        self._device = 'cuda'

    def get_random_image_pair(self):
        """
        Gets a pair of random images for different scenes of the same object
        """
        ep_a_data = self._dataset[randrange(len(self._dataset))]
        ep_b_data = self._dataset[randrange(len(self._dataset))]

        # ep_a_name = ep_a_data['epsiode_name']
        # ep_a_idx = ep_a_data['idx_a']
        # ep_b_name = ep_b_data['epsiode_name']
        # ep_b_idx = ep_b_data['idx_b']
        # print(f'IMG A from {ep_a_name}: {ep_a_idx}; IMG B from {ep_b_name}: {ep_b_idx}')

        return ep_a_data, ep_b_data

    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        data_a, data_b = self.get_random_image_pair()

        self._rgb_a = np.copy(data_a['data_a']['rgb'])
        self._rgb_b = np.copy(data_b['data_b']['rgb'])

        rgb_tensor_a = data_a['data_a']['rgb_tensor'].to(self._device).unsqueeze(0)
        rgb_tensor_b = data_b['data_b']['rgb_tensor'].to(self._device).unsqueeze(0)

        # predict best uv_b for uv_a
        with torch.no_grad():
            # now localize the correspondences
            # push rgb_a, rgb_b through the network
            rgb_tensor_stack = torch.cat([rgb_tensor_a, rgb_tensor_b])
            out = self._model.forward(rgb_tensor_stack)
            des_img_stack = out['descriptor_image']
            self._des_img_a = des_img_stack[:1]
            self._des_img_b = des_img_stack[1:]

        self.find_best_match(None, 0, 0, None, None)

    def find_best_match(self, event, u, v, flags, param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """
        if self._paused:
            return

        uv = torch.ones([2, 1], dtype=torch.int64, device=self._device)
        uv[0, 0] = u
        uv[1, 0] = v

        # Show img a
        img_1_with_reticle = cv2.cvtColor(self._rgb_a, cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticles(
            img_1_with_reticle, uv[0, :], uv[1, :], [0, 255, 0])
        cv2.imshow("source", img_1_with_reticle)

        # Get descriptor a
        des_a = pdc_utils.index_into_batch_image_tensor(
            self._des_img_a, uv.unsqueeze(0)).permute([0, 2, 1])

        # Make heatmap
        W = self._rgb_a.shape[1]
        H = self._rgb_a.shape[0]
        diag = np.sqrt(W**2 + H**2)
        sigma = self._sigma * diag
        heatmap = loss_utils.compute_heatmap_from_descriptors(
            des_a, self._des_img_b, sigma=sigma, type=self._heatmap_type)
        heatmap = heatmap.squeeze(dim=0)

        # Find match.
        best_uv_b = predict.get_integral_preds_2d(heatmap)
        best_uv_b = torch.transpose(best_uv_b, 0, 1)

        # Show img b
        img_2_with_reticle = cv2.cvtColor(self._rgb_b, cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticles(
            img_2_with_reticle, best_uv_b[0, :], best_uv_b[1, :], [0, 255, 0])
        cv2.imshow("target", img_2_with_reticle)

        # Show heatmap
        heatmap_k = heatmap[0].detach().cpu().numpy() # [H, W]
        heatmap_k = heatmap_k / np.sum(heatmap_k)
        heatmap_rgb = vis_utils.colormap_from_heatmap(heatmap_k, normalize=True)
        heatmap_rgb = (heatmap_rgb * 0.6 + self._rgb_b * 0.4).astype(np.uint8)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticles(
            heatmap_rgb, best_uv_b[0, :], best_uv_b[1, :], [0, 255, 0])
        cv2.imshow("heatmap", heatmap_rgb)

    def run(self):
        self._get_new_images()
        #cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        self._get_new_images()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._get_new_images()
            elif k == ord('p'):
                if self._paused:
                    print("un pausing")
                    self._paused = False
                else:
                    print("pausing")
                    self._paused = True


if __name__ == "__main__":
    # TODO(sfeng): make this better...
    config_file = os.path.join(
        getDenseCorrespondenceSourceDir(),
        'config/dense_correspondence/global/drake_sim_dynamic.yaml')
    episode_file = os.path.join(
        os.path.join(getDenseCorrespondenceSourceDir(),
        'config/dense_correspondence/dataset/single_object/tea_bottle_episodes.yaml'))
    model_file = os.path.join(
        os.getenv("DATA_ROOT"), 'pdc/dev/experiments/heatmap/trained_models',
        '2020-02-17-17-28-47_resnet50__dataset_tea_bottle',
        'net_dy_epoch_5_iter_0_model.pth')

    heatmap_vis = HeatmapVisualization(config_file, episode_file, model_file)
    print("starting heatmap vis")
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
