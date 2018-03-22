#!/usr/bin/python

import sys, os
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
from spartan_dataset_masked import SpartanDataset

import dense_correspondence_manipulation.utils.utils as utils


def test():
    #res_a = forward_on_img(last_net[0], img_a_rgb)
    #res_b = forward_on_img(last_net[0], img_b_rgb)

    for i in range(100):
        data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = lf[i]

        img_a = Variable(img_a.cuda(), requires_grad=False)
        img_b = Variable(img_b.cuda(), requires_grad=False)

        W = 640
        H = 480
        N = 1

        if data_type == "matches":
            matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
            matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
            non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
            non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)


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

    def forward_on_img(self, img):
        """
        Runs the network forward on an image
        :param img: img is an image as a numpy array in opencv format
        :return:
        """
        img = DenseCorrespondenceNetwork.IMAGE_TO_TENSOR(img)
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
    def test(nn, dataset, data_idx=1):
        data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = dataset[i]

        img_a = Variable(img_a.cuda(), requires_grad=False)
        img_b = Variable(img_b.cuda(), requires_grad=False)

        W = 640
        H = 480
        N = 1

        if data_type == "matches":
            matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
            matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
            non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
            non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)


def run():
    pass

def main(config):
    eval = DenseCorrespondenceEvaluation(config)
    eval.load_network_from_config("10_scenes_drill")
    test_dataset = SpartanDataset(mode="test")

def test():
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'evaluation.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    default_config = utils.get_defaults_config()
    utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

    main(config)

if __name__ == "__main__":
    test()