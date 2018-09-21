#!/usr/bin/python

import sys, os
import numpy as np
import logging
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()


from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset



class ConvolutionalSpatialTransformer(nn.Module):

    def __init__(self, image_height, image_width, grid_size):
        self._image_height = image_height
        self._image_width = image_width
        self._grid_size = grid_size # s
        self._construct_grids()

    def _construct_grid(self):
        """
        Makes two grids each of shape [s*H, s*W,
        G_0 is [
        :return:
        :rtype:
        """

    @staticmethod
    def dense_grid(s, shape):
        """
        shape is [N, C, H, W]

        Generates grid of size [N, s*H, s*W, 2]

        G_0 = just repeats the original coordinates into s x s grids
        G = repeats an s x s pattern
        :param x:
        :type x:
        :return: [G_0, G]
        :rtype:
        """



        height = shape[2]
        width = shape[3]
        H = height
        W = width
        N = shape[0]

        # construct G
        grid = np.zeros([s, s, 2], dtype=np.float32)
        grid[:, :, 0] = np.expand_dims(
            np.repeat(np.expand_dims(np.linspace(-s * 1.0 / height, s * 1.0 / height, s), 0), repeats=s, axis=0).T, 0)
        grid[:, :, 1] = np.expand_dims(
            np.repeat(np.expand_dims(np.linspace(-s * 1.0 / width, s * 1.0 / width, s), 0), repeats=s, axis=0), 0)

        # grid is s x s x 2

        # (s*H) x (s*W) x 2
        grid_dense = np.tile(grid, [height, width, 1])

        # N x (s*H) x (s*W) x 2
        G = np.repeat(np.expand_dims(grid_dense, 0), repeats=N, axis=0)

        # constructing G_0
        G_0 = np.zeros([N, H, W, 2], dtype=np.float32)


        xy = np.zeros([H,W,2])
        xy[:,:,0] = np.expand_dims(
            np.repeat(np.expand_dims(np.linspace(-1,1,H), 0), repeats=H, axis=0).T, 0)
        xy[:, :, 1] = np.expand_dims(
            np.repeat(np.expand_dims(np.linspace(-1, 1, W), 0), repeats=W, axis=0), 0)


        xy_resize = np.repeat(xy, s, axis=0)
        xy_resize = np.repeat(xy, s, axis=1)

        # now xy_resize has shape [s*H, s*W, 2]
        # finally need to expand the first dimension
        G_0 = np.repeat(np.expand_dims(xy_resize, 0), repeats=N, axis=0)

        return G_0, G





