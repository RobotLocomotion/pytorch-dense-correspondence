#!/usr/bin/python

import sys, os
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()

import dense_correspondence as DC
sys.path.insert(0, '../../pytorch-segmentation-detection/vision/')
sys.path.append('../../pytorch-segmentation-detection/')


from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated

import numpy as np
import glob

import sys; sys.path.append('../dataset')
sys.path.append('../correspondence_tools')
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder



class DenseCorrespondenceNetwork(object):

    IMAGE_TO_TENSOR = valid_transform = transforms.Compose([transforms.ToTensor(), ])

    def __init__(self, fcn, descriptor_dimension, image_width=640,
                 image_height=480):

        self._fcn = fcn
        self._descriptor_dimension = descriptor_dimension
        self._image_width = image_width
        self._image_height = image_height

    @property
    def fcn(self):
        return self._fcn

    @property
    def descriptor_dimension(self):
        return self._descriptor_dimension

    @property
    def image_shape(self):
        return [self._image_height, self._image_width]

    def forward_on_img(self, img, cuda=True):
        """
        Runs the network forward on an image
        :param img: img is an image as a numpy array in opencv format [0,255]
        :return:
        """
        img_tensor = DenseCorrespondenceNetwork.IMAGE_TO_TENSOR(img)

        if cuda:
            img_tensor.cuda()

        return self.forward_on_img_tensor(img_tensor)


    def forward_on_img_tensor(self, img):
        """
        Runs the network forward on an img_tensor
        :param img: (C x H X W) in range [0.0, 1.0]
        :return:
        """
        img = img.unsqueeze(0)
        img = Variable(img.cuda())
        res = self.fcn(img)
        res = res.squeeze(0)
        res = res.permute(1, 2, 0)
        res = res.data.cpu().numpy().squeeze()

        return res


    @staticmethod
    def from_config(config):
        """
        Load a network from a config file

        :param config: Dict specifying details of the network architecture
        e.g.
            path_to_network: /home/manuelli/code/dense_correspondence/recipes/trained_models/10_drill_long_3d
            parameter_file: dense_resnet_34_8s_03505.pth
            descriptor_dimensionality: 3
            image_width: 640
            image_height: 480

        :return:
        """

        fcn = resnet_dilated.Resnet34_8s(num_classes=config['descriptor_dimension'])
        fcn.load_state_dict(torch.load(config['path_to_network_params']))
        fcn.cuda()
        fcn.eval()

        return DenseCorrespondenceNetwork(fcn, config['descriptor_dimension'],
                                          image_width=config['image_width'],
                                          image_height=config['image_height'])

    @staticmethod
    def find_best_match(pixel_a, res_a, res_b):
        """
        Compute the correspondences between the pixel_a location in image_a
        and image_b

        :param pixel_a: vector of (x,y) pixel coordinates
        :param res_a: array of dense descriptors
        :param res_b: array of dense descriptors
        :param pixel_b: Ground truth . . .
        :return: (best_match_idx, best_match_diff, norm_diffs)
        """

        debug = True

        descriptor_at_pixel = res_a[pixel_a[0], pixel_a[1]]
        height, width, _ = res_a.shape



        if debug:
            print "height: ", height
            print "width: ", width
            print "res_b.shape: ", res_b.shape


        # non-vectorized version
        # norm_diffs = np.zeros([height, width])
        # for i in xrange(0, height):
        #     for j in xrange(0, width):
        #         norm_diffs[i,j] = np.linalg.norm(res_b[i,j] - descriptor_at_pixel)**2

        norm_diffs = np.sum(np.square(res_b - descriptor_at_pixel), axis=2)

        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_idx = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_idx]

        return best_match_idx, best_match_diff, norm_diffs



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
    def test(dcn, dataset, data_idx=1, visualize=False):
        """

        :param dcn: DenseCorrespondenceNetwork
        :param dataset: DenseCorrespondenceDataset
        :param data_idx: idx of data point to test
        :return:
        """

        # data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = dataset[data_idx]

        scene_name = '13_drill_long_downsampled'
        img_idx_a = utils.getPaddedString(0)
        img_idx_b = utils.getPaddedString(737)

        rgb_a = dataset.get_rgb_image_from_scene_name_and_idx(scene_name, img_idx_a)
        depth_a = dataset.get_depth_image_from_scene_name_and_idx(scene_name, img_idx_a)
        mask_a = dataset.get_mask_image_from_scene_name_and_idx(scene_name, img_idx_a)
        pose_a = dataset.get_pose_from_scene_name_and_idx(scene_name, img_idx_a)

        rgb_b = dataset.get_rgb_image_from_scene_name_and_idx(scene_name, img_idx_b)
        depth_b = dataset.get_depth_image_from_scene_name_and_idx(scene_name, img_idx_b)
        mask_b = dataset.get_mask_image_from_scene_name_and_idx(scene_name, img_idx_b)
        pose_b = dataset.get_pose_from_scene_name_and_idx(scene_name, img_idx_b)


        # find correspondences
        num_attempts = 20
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(depth_a, pose_a,
                                                                            depth_b, pose_b,
                                                                            num_attempts=num_attempts,
                                                                            img_a_mask=mask_a)

        # if data_type == "matches":
        #     # why are we converting everything to variables here?
        #     matches_a = matches_a.cuda().squeeze(0)
        #     matches_b = matches_b.cuda().squeeze(0)
        #     non_matches_a = non_matches_a.cuda().squeeze(0)
        #     non_matches_b = non_matches_b.cuda().squeeze(0)
        #
        # # run this through the network
        # print "img_a.shape: ", img_a.shape
        # print "matches_a.shape: ", matches_a.shape
        #
        #
        # res_a = dcn.forward_on_img_tensor(img_a)
        # res_b = dcn.forward_on_img_tensor(img_b)
        #

        res_a = dcn.forward_on_img(rgb_a)
        res_b = dcn.forward_on_img(rgb_b)

        print "res_a.shape: ", res_a.shape

        test_idx = 4
        pixel_a = [int(uv_a[0][test_idx]), int(uv_a[1][test_idx])]
        pixel_b = [int(uv_b[0][test_idx]), int(uv_b[1][test_idx])]

        print "pixel_a: ", pixel_a
        best_match_idx, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a, res_b)

        if visualize:
            (fig, axes) = correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b, pixel_a,
                                                               pixel_b, circ_color='g', show=False)
            correspondence_plotter.plot_correspondences_direct(rgb_a, depth_a, rgb_b, depth_b, pixel_a, best_match_idx, circ_color='b', use_previous_plot=(fig, axes), show=True)


        # img_a_descriptors = dcn.forward_on_img(img_a)

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