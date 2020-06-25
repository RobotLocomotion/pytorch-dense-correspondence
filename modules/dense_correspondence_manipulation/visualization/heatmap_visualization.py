import os
import cv2
import numpy as np
from random import randrange
import random
from PIL import Image
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
                 camera_list=None, # (optional) list of cameras you are ok getting images from
                 verbose=False,
                 target_camera_names=None,
                 sample_same_episode=False,
                 display_confidence_value=True,
                 use_custom_target_image_func=False,
                 ):
        self._dataset = dataset
        self._model = model
        self._config = config
        self._rescale_heatmap_for_vis = False
        self._visualize_3D = visualize_3D
        self._meshcat_vis = None # to be populated as necessary
        self._camera_list = camera_list
        self._verbose = verbose
        self._target_camera_names = target_camera_names
        self._sample_same_episode = sample_same_episode
        self._display_confidence_value = display_confidence_value
        self._use_custom_target_image_func = use_custom_target_image_func


        self._image_save_dict = dict()


        if self._visualize_3D:
            self.initialize_meshcat()

        self._paused = False
        self._sigma = self._config['loss_function']['heatmap']['sigma_fraction']
        self._heatmap_type = self._config['loss_function']['heatmap']['heatmap_type']
        self._device = 'cuda'

        # store data for each image
        self._data_a = None
        self._data_b = None

        self._target_data = []

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
        data = None
        for i in range(100): # try 100 times to get image from camera you want
            data = self._dataset[randrange(len(self._dataset))]['data_a']
            if self._camera_list is None:
                break
            elif data['camera_name'] in self._camera_list:
                break

        if data is None:
            raise ValueError("couldn't get an image from camera you specified")

        return data

    def get_random_image_pair(self):
        data = None
        for i in range(100):  # try 100 times to get image from camera you want
            data = self._dataset[randrange(len(self._dataset))]
            if self._camera_list is None:
                break

        if data is None:
            raise ValueError("couldn't get an image from camera you specified")

        return data

    def _get_new_target_images(self):
        """
        Populate the self._target_data dict
        :return:
        :rtype:
        """
        data = self.get_random_image()
        # print("data.keys()", data.keys())

        if self._sample_same_episode:
            data = self._data_a

        episode_name = data['episode_name']
        episode = self._dataset.episodes[episode_name]
        idx = data['idx']



        if self._sample_same_episode:
            idx = random.randint(0, episode.length-1)
            # idx = data['idx']

        # print("idx_a", data['idx'])
        # print("episode.length", episode.length)
        # print("idx_b", idx)

        camera_names = None
        if self._target_camera_names is not None:
            camera_names = self._target_camera_names
        else:
            camera_names = [random.choice(episode.camera_names)]

        self._target_data = []
        for i, camera_name in enumerate(camera_names):
            data = episode.get_image_data(camera_name, idx)
            data['rgb_tensor'] = self._dataset._rgb_image_to_tensor(data['rgb'])
            self._target_data.append(data)


    def _get_multiple_different_target_images(self, num_images=2):
        self._target_data = []
        for i in range(num_images):
            data = self.get_random_image()
            data['rgb_tensor'] = self._dataset._rgb_image_to_tensor(data['rgb'])
            self._target_data.append(data)



    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        self._data_a = self.get_random_image()
        # self._data_b = self._get_new_target_images()

        if self._use_custom_target_image_func:
            self._get_multiple_different_target_images()
        else:
            self._get_new_target_images()


        if self._verbose:
            def print_image_data(data):
                print("camera_name:", data['camera_name'])
                print("episode_name:", data['episode_name'])
                print("idx:", data['idx'])

            print("\n\n-----------------")
            print("Source Image Data:")
            print_image_data(self._data_a)

            # print("\nTarget Image Data")
            # print_image_data(self._data_b)

        self._compute_descriptor_images()

    def _swap_images(self):
        """
        Swap the a and b images
        :return:
        :rtype:
        """
        data_a = self._data_a
        data_a = self._target_data[0]
        self._target_data = [data_a]
        self._data_a = data_a

        self._compute_descriptor_images()


    def _compute_descriptor_images(self):
        """
        Compute the descriptor images
        :return:
        :rtype:
        """

        rgb_tensor_list = []
        rgb_tensor_list.append(self._data_a['rgb_tensor'])

        for data in self._target_data:
            rgb_tensor_list.append(data['rgb_tensor'])

        rgb_tensor_batch = torch.stack(rgb_tensor_list).to(self._device)

        # predict best uv_b for uv_a
        with torch.no_grad():
            # push rgb_tensor_batch through the network
            out = self._model.forward(rgb_tensor_batch)
            des_img_batch = out['descriptor_image']

            self._data_a['descriptor_image'] = des_img_batch[0]

            for counter, data in enumerate(self._target_data):
                data['descriptor_image'] = des_img_batch[counter+1]


        self.mouse_callback(None, 0, 0, None, None)

    def find_best_match(self, u, v, source_data, target_data, img_suffix):

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
        img_1_with_reticle = cv2.cvtColor(np.copy(source_data['rgb']), cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticle(img_1_with_reticle, u, v, [0, 255, 0])
        cv2.imshow("source", img_1_with_reticle)

        self._image_save_dict['source'] = img_1_with_reticle

        # Get descriptor a
        # print("source_data['descriptor_image'].shape", source_data['descriptor_image'].shape)
        des_a = pdc_utils.index_into_batch_image_tensor(
            source_data['descriptor_image'].unsqueeze(0), uv.unsqueeze(0)).permute([0, 2, 1])

        # spatial expectation
        spatial_pred = predict.get_spatial_expectation(des_a,
                                                       target_data['descriptor_image'],
                                                       sigma=self._config['network']['sigma_descriptor_heatmap'],
                                                       type='exp', return_heatmap=True)

        # [H, W]
        heatmap = spatial_pred['heatmap_no_batch'].squeeze()

        # [2,]
        uv_spatial_pred = spatial_pred['uv'].squeeze()


        uv_spatial_pred = uv_spatial_pred.squeeze()
        des_b = target_data['descriptor_image'][:, int(uv_spatial_pred[1]), int(uv_spatial_pred[0])]
        heatmap_value = heatmap[int(uv_spatial_pred[1]), int(uv_spatial_pred[0])]

        # distance in descriptor space
        descriptor_distance = torch.norm(des_a - des_b).item()


        # Find match.

        # Show img b
        H, W = heatmap.shape
        img_2_with_reticle = cv2.cvtColor(np.copy(target_data['rgb']), cv2.COLOR_RGB2BGR)
        vis_utils.draw_reticle(
            img_2_with_reticle, uv_spatial_pred[0], uv_spatial_pred[1], [0, 255, 0])

        # print descriptor distance in bottom left of image
        org = (int(0.05*W), int(0.95*H))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255) # white
        thickness = 2 # pixels

        # text = "%.2f" %(descriptor_distance)
        if self._display_confidence_value:
            text = "%.2f" %(heatmap_value)
            cv2.putText(img_2_with_reticle, text, org, font, fontScale, color, thickness)
        cv2.imshow("target_" + img_suffix, img_2_with_reticle)

        key = "target_" + img_suffix
        self._image_save_dict[key] = img_2_with_reticle


        # show heatmap
        heatmap_k = heatmap.detach().cpu().numpy()  # [H, W]
        heatmap_rgb = vis_utils.colormap_from_heatmap(heatmap_k, normalize=self._rescale_heatmap_for_vis)
        heatmap_blend = (heatmap_rgb * 0.6 + target_data['rgb'] * 0.4).astype(np.uint8)  # blend
        heatmap_blend = cv2.cvtColor(heatmap_blend, cv2.COLOR_RGB2BGR)
        heatmap_blend_wr = np.copy(heatmap_blend)
        vis_utils.draw_reticle(
            heatmap_blend_wr, uv_spatial_pred[0], uv_spatial_pred[1], [0, 255, 0])

        if self._display_confidence_value:
            cv2.putText(heatmap_blend_wr, text, org, font, fontScale, color, thickness)
        cv2.imshow("heatmap_" + img_suffix, heatmap_blend_wr)

        key = "heatmap_" + img_suffix
        self._image_save_dict[key] = heatmap_blend_wr


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


    def mouse_callback(self, event, u, v, flags, param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """
        if self._paused:
            return

        self._image_save_dict = dict()

        for i, data in enumerate(self._target_data):
            self.find_best_match(u, v, self._data_a, data, img_suffix=str(i))

    def _save_images(self):
        output_dir = os.path.join(getDenseCorrespondenceSourceDir(),
                                  'sandbox/heatmap_visualization',
                                  pdc_utils.get_current_YYYY_MM_DD_hh_mm_ss(),
                                  )


        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)


        print("\n\nsaving images to: ", output_dir)
        for key, img in self._image_save_dict.items():
            filename = os.path.join(output_dir, "%s.png" %(key))
            cv2.imwrite(filename, img) # it's already in BGR format
            # img_PIL = Image.fromarray(img)
            # img_PIL.save(filename)

    def run(self):
        self._get_new_images()
        #cv2.namedWindow('target')
        # cv2.namedWindow('source')
        # cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.mouse_callback)

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
            elif k == ord('t'): # get new target image
                if self._use_custom_target_image_func:
                    self._get_multiple_different_target_images()
                else:
                    self._get_new_target_images()
                self._compute_descriptor_images()
            elif k == ord('f'):
                self._save_images()
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
