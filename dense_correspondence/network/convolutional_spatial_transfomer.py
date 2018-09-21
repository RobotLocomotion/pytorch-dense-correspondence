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
import torch.nn.functional as F
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset


"""
See Choy et al, 2016, Universal Correspondence Network, for reference
"""
class ConvolutionalSpatialTransformer(nn.Module):

    def __init__(self, image_height, image_width, grid_size):
        super(ConvolutionalSpatialTransformer, self).__init__()
        self._image_height = image_height
        self._image_width = image_width
        self._grid_size = grid_size # s
        self._construct_grid()

        ## These conv layers expect input of for example shape: [1,3,480,640]
        ## And given shape above would output: [1,4,480,640]
        ##     - where the 4 is the number of parameters in the scale+rotation
        self.conv = nn.Sequential( 
            torch.nn.Conv2d(3, 4, 5, stride=1, padding=2),
            torch.nn.Conv2d(4, 4, 5, stride=1, padding=2),
            torch.nn.Conv2d(4, 4, 5, stride=1, padding=2),
            torch.nn.Conv2d(4, 4, 5, stride=1, padding=2),
        )

        self.reduce_conv = nn.Sequential(
            torch.nn.Conv2d(3,3,self._grid_size, self._grid_size)
        )

    def _construct_grid(self):
        """
        Makes two grids each of shape [s*H, s*W,
        G_0 is [
        :return:
        :rtype:
        """
        self._G0, self._G =  ConvolutionalSpatialTransformer.dense_grid(self._grid_size, [1, 3, self._image_height, self._image_width])
        self._G0 = Variable(torch.from_numpy(self._G0),requires_grad=False)
        self._G = Variable(torch.from_numpy(self._G),requires_grad=False)

    
    def repeat_tile(self, tensor, repeat_size):
        """
        Takes for example a tensor of shape [N, 4, 480, 640]
        And makes it a tensor of shape [N, 4, 480*s, 640*s]
        Following the pattern best diagrammed:

        input[0,0,:,:]: a b
                        c d 

        output[0,0,:,:]: a a b b
                         a a b b
                         c c d d
                         c c d d

        This is opposed to the repeated_tensor intermediate below, which has the following pattern:

        repeated_tensor[0,0,:,:]:  a b a b
                                   c d c d
                                   a b a b
                                   c d c d

        """
        [N, C, H, W] = tensor.shape
        repeated_tensor = tensor.repeat(1,1,repeat_size,repeat_size)
        return torch.transpose(repeated_tensor.view(N,C, repeat_size,-1), 2,3).contiguous().view(N,C,repeat_size*H,repeat_size*W)

    def forward(self, x):
        theta_full_resolution = self.conv(x)                                      # this is shape   N,C,  H  ,W
        theta_repeated = self.repeat_tile(theta_full_resolution, self._grid_size) # this is shape   N,C,  H*s,W*s
        theta_repeated = theta_repeated.permute(0,2,3,1)                          # change to shape N,H*s,W*s,C

        # now perform the matrix multiplication
        G_transformed_top    = (theta_repeated[:,:,:,0:2]*self._G).sum(3).unsqueeze(3)
        G_transformed_bottom = (theta_repeated[:,:,:,2: ]*self._G).sum(3).unsqueeze(3)
        G_transformed = torch.cat([G_transformed_top,G_transformed_bottom], 3)

        final_grid = self._G0 + G_transformed
        x = F.grid_sample(x, final_grid)
        x = self.reduce_conv(x)
        return x


    @staticmethod
    def dense_grid(s, shape):
        """
        Generates grid of size [N, s*H, s*W, 2]

        G_0 = just repeats the original coordinates into s x s grids
        G = repeats an s x s pattern
        :param s: repeat size
        :type s: int
        :param shape: follows shape is [N, C, H, W]
        :type shape: list of ints
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
            np.repeat(np.expand_dims(np.linspace(-1,1,H), 0), repeats=W, axis=0).T, 0)
        xy[:, :, 1] = np.expand_dims(
            np.repeat(np.expand_dims(np.linspace(-1, 1, W), 0), repeats=H, axis=0), 0)


        xy_resize = np.repeat(xy, s, axis=0)
        xy_resize = np.repeat(xy_resize, s, axis=1)

        # now xy_resize has shape [s*H, s*W, 2]
        # finally need to expand the first dimension
        G_0 = np.repeat(np.expand_dims(xy_resize, 0), repeats=N, axis=0)

        return G_0.astype(np.float32), G.astype(np.float32)





