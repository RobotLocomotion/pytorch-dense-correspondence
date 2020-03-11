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
from dense_correspondence_manipulation.utils import constants
from dense_correspondence_manipulation.utils import meshcat_utils

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

    def __init__(self,
                 config,
                 dataset,
                 model,
                 rescale_heatmap_for_vis=False, # peak of heatmap is normalized to be one for visualization
                 visualize_3D=False,
                 ):
        self._dataset = dataset
        self._model = model
        self._config = config
        self._rescale_heatmap_for_vis = False
        self._visualize_3D = visualize_3D
        self._meshcat_vis = None # to be populated as necessary

        if self._visualize_3D:
            self.initialize_meshcat()

        self._paused = False
        self._sigma = self._config['loss_function']['heatmap']['sigma_fraction']
        self._heatmap_type = self._config['loss_function']['heatmap']['heatmap_type']
        self._device = 'cuda'

        # store data for each image
        self._data_a = None
        self._data_b = None

    def initialize_meshcat(self):
        """
        Initializes the meshcat visualizer
        :return:
        :rtype:
        """
        import meshcat
        import meshcat.geometry as meshcat_geom
        import meshcat.transformations as meshcat_tf

        self._meshcat_vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")


    def set_data_a(self, data):
        """
        Set the image data for image a
        :param data:
        :type data:
        :return:
        :rtype:
        """
        self._data_a = data

    def set_data_b(self, data):
        """
        Set image data for image b
        :param data:
        :type data:
        :return:
        :rtype:
        """
        self._data_b = data

    def get_random_image(self):
        """
        Return a random image from the dataset
        :return:
        :rtype:
        """
        data = self._dataset[randrange(len(self._dataset))]['data_a']
        return data

    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        self._data_a = self.get_random_image()
        self._data_b = self.get_random_image()
        self._compute_descriptor_images()

    def _swap_images(self):
        """
        Swap the a and b images
        :return:
        :rtype:
        """
        data_a = self._data_a
        data_b = self._data_b

        self._data_a = data_b
        self._data_b = data_a

        self._compute_descriptor_images()


    def _compute_descriptor_images(self):
        """
        Compute the descriptor images
        :return:
        :rtype:
        """

        data_a = self._data_a
        data_b = self._data_b
        self._rgb_a = np.copy(data_a['rgb'])
        self._rgb_b = np.copy(data_b['rgb'])

        rgb_tensor_a = self._data_a['rgb_tensor'].to(self._device).unsqueeze(0)
        rgb_tensor_b = self._data_b['rgb_tensor'].to(self._device).unsqueeze(0)

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

        # [2,1] = [2, N] with N = 1
        uv = torch.ones([2, 1], dtype=torch.int64, device=self._device)
        uv[0, 0] = u
        uv[1, 0] = v

        # Show img a
        img_1_with_reticle = cv2.cvtColor(self._rgb_a, cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticle(img_1_with_reticle, u, v, [0, 255, 0])
        cv2.imshow("source", img_1_with_reticle)

        # Get descriptor a
        des_a = pdc_utils.index_into_batch_image_tensor(
            self._des_img_a, uv.unsqueeze(0)).permute([0, 2, 1])

        # spatial expectation
        spatial_pred = predict.get_spatial_expectation(des_a,
                                                       self._des_img_b,
                                                       sigma=self._config['network']['sigma_descriptor_heatmap'],
                                                       type='exp', return_heatmap=True)

        # [H, W]
        heatmap = spatial_pred['heatmap_no_batch'].squeeze()

        # [2,]
        uv_spatial_pred = spatial_pred['uv'].squeeze()


        uv_spatial_pred = uv_spatial_pred.squeeze()
        des_b = self._des_img_b[0,:, int(uv_spatial_pred[1]), int(uv_spatial_pred[0])]
        heatmap_value = heatmap[int(uv_spatial_pred[1]), int(uv_spatial_pred[0])]

        # distance in descriptor space
        descriptor_distance = torch.norm(des_a - des_b).item()


        # Find match.

        # Show img b
        H, W = heatmap.shape
        img_2_with_reticle = cv2.cvtColor(self._rgb_b, cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticle(
            img_2_with_reticle, uv_spatial_pred[0], uv_spatial_pred[1], [0, 255, 0])

        # print descriptor distance in bottom left of image
        org = (int(0.05*W), int(0.95*H))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255) # white
        thickness = 2 # pixels

        # text = "%.2f" %(descriptor_distance)
        text = "%.2f" %(heatmap_value)
        cv2.putText(img_2_with_reticle, text, org, font, fontScale, color, thickness)
        cv2.imshow("target", img_2_with_reticle)


        # show heatmap
        heatmap_k = heatmap.detach().cpu().numpy()  # [H, W]
        heatmap_rgb = vis_utils.colormap_from_heatmap(heatmap_k, normalize=self._rescale_heatmap_for_vis)
        heatmap_blend = (heatmap_rgb * 0.6 + self._rgb_b * 0.4).astype(np.uint8)  # blend
        heatmap_blend = cv2.cvtColor(heatmap_blend, cv2.COLOR_RGB2BGR)
        heatmap_blend_wr = np.copy(heatmap_blend)
        vis_utils.draw_reticle(
            heatmap_blend_wr, uv_spatial_pred[0], uv_spatial_pred[1], [0, 255, 0])
        cv2.putText(heatmap_blend_wr, text, org, font, fontScale, color, thickness)
        cv2.imshow("heatmap", heatmap_blend_wr)


        if self._visualize_3D:

            # do 3D prediction
            # this depth image is already in meters . . .
            depth_img_b = self._data_b['depth_int16'] / constants.DEPTH_IM_SCALE

            # [1, H, W], N = 1
            depth_img_b = torch.Tensor(depth_img_b).to(self._device).unsqueeze(0)

            spatial_pred_3D = predict.get_integral_preds_3d(heatmap.unsqueeze(0),
                                                            depth_img_b,
                                                            compute_uv=False)

            # Predicted uv_b
            # [1, 2]
            uv_spatial_pred_np = uv_spatial_pred.unsqueeze(0).cpu().numpy()
            depth_pred = spatial_pred_3D['z'].cpu().numpy()
            pts_pred = pdc_utils.pinhole_unprojection(uv_spatial_pred_np, depth_pred, self._data_b['K'])


            # clear visualizer
            self._meshcat_vis.delete()
            meshcat_utils.visualize_points(self._meshcat_vis, 'spatial_prediction', pts_pred, color=[1, 0, 0], size=0.01, T=self._data_b['T_world_camera'])
            #
            #
            # # visualize pointclouds from data_a, data_b
            # meshcat_utils.visualize_pointcloud(self._meshcat_vis, 'pointcloud_a', self._data_a['depth_int16']/constants.DEPTH_IM_SCALE, K=self._data_a['K'], rgb=self._data_a['rgb'], T_world_camera=self._data_a['T_world_camera'])
            #
            # meshcat_utils.visualize_pointcloud(self._meshcat_vis, 'pointcloud_b',
            #                                    self._data_b['depth_int16'] / constants.DEPTH_IM_SCALE,
            #                                    K=self._data_b['K'], rgb=self._data_b['rgb'],
            #                                    T_world_camera=self._data_b['T_world_camera'])

            # visualize heatmap pointcloud
            heatmap_blend_rgb = (heatmap_rgb * 0.6 + self._rgb_b * 0.4).astype(np.uint8)
            meshcat_utils.visualize_pointcloud(self._meshcat_vis, 'heatmap',
                                               self._data_b['depth_int16'] / constants.DEPTH_IM_SCALE,
                                               K=self._data_b['K'], rgb=heatmap_blend_rgb,
                                               T_world_camera=self._data_b['T_world_camera'])

    def run(self):
        self._get_new_images()
        #cv2.namedWindow('target')
        # cv2.namedWindow('source')
        # cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        self._get_new_images()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._get_new_images()
            elif k == ord('s'):
                # swap a and b images
                self._swap_images()
            elif k == ord('a'):
                self._data_a = self.get_random_image()
                self._compute_descriptor_images()
            elif k == ord('q'):
                # get new data_a with idx 0
                pass
            elif k == ord('g'):
                # get new data_b with idx 0
                pass
            elif k == ord('b'):
                self._data_b = self.get_random_image()
                self._compute_descriptor_images()
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

    dataset = load_dataset(episode_config=episode_file, pdc_config=config_file)
    model = load_model(model_file)



    heatmap_vis = HeatmapVisualization(config_file, episode_file, model_file)
    print("starting heatmap vis")
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
