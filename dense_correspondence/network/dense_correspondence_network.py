#!/usr/bin/python

import sys, os
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()


from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated

import numpy as np


class DenseCorrespondenceNetwork(torch.nn.Module):

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


    def parameters(self):
        """
        :return: Parameters of the fcn to be adjusted during training
        :rtype: ?
        """
        return self.fcn.parameters()

    def state_dict(self):
        """
        Gets the state_dict for the network
        :return:
        :rtype:
        """
        return self.fcn.state_dict()

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

    def forward(self, img):
        """
        Simple forward pass on the network
        :param img: input tensor img.shape = [N,descriptor_dim, H , W] where
                    N is the batch size
        :type img: torch.Variable or torch.Tensor
        :return: same as input type
        :rtype:
        """

        return self.fcn(img)

    def process_network_output(self, image_pred, N):
        """
        Processes the network output into a new shape

        :param image_pred: output of the network img.shape = [N,descriptor_dim, H , W]
        :type image_pred: torch.Tensor
        :param N: batch size
        :type N: int
        :return: same as input, new shape is [N, W*H, descriptor_dim]
        :rtype:
        """

        W = self._image_width
        H = self._image_height
        image_pred = image_pred.view(N, self.descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred


    @staticmethod
    def from_config(config, load_stored_params=True):
        """
        Load a network from a config file

        :param load_stored_params: whether or not to load stored params, if so there should be
            a "path_to_network" entry in the config
        :type load_stored_params: bool

        :param config: Dict specifying details of the network architecture

        e.g.
            path_to_network: /home/manuelli/code/dense_correspondence/recipes/trained_models/10_drill_long_3d
            parameter_file: dense_resnet_34_8s_03505.pth
            descriptor_dimensionality: 3
            image_width: 640
            image_height: 480

        :return: DenseCorrespondenceNetwork
        :rtype:
        """

        fcn = resnet_dilated.Resnet34_8s(num_classes=config['descriptor_dimension'])

        if load_stored_params:
            path_to_network_params = utils.convert_to_absolute_path(config['path_to_network_params'])
            fcn.load_state_dict(torch.load(path_to_network_params))
            fcn.cuda()
            fcn.eval()
        else:
            fcn.cuda()
            fcn.train()



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

        debug = False

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