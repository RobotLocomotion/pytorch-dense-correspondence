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

class NetworkMode:
    TRAIN = 0 # don't normalize images
    TEST = 1 # normalize images

class DenseCorrespondenceNetwork(object):

    IMAGE_TO_TENSOR = valid_transform = transforms.Compose([transforms.ToTensor(), ])

    def __init__(self, fcn, descriptor_dimension, image_width=640,
                 image_height=480):

        self._fcn = fcn
        self._descriptor_dimension = descriptor_dimension
        self._image_width = image_width
        self._image_height = image_height

        # this defaults to the identity transform
        self._image_mean = np.zeros(3)
        self._image_std_dev = np.ones(3)

        # defaults to no image normalization, assume it is done by dataset loader instead
        self.mode = NetworkMode.TRAIN
        self.config = dict()


    @property
    def fcn(self):
        return self._fcn

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def descriptor_dimension(self):
        return self._descriptor_dimension

    @property
    def image_shape(self):
        return [self._image_height, self._image_width]

    @property
    def image_mean(self):
        return self._image_mean

    @image_mean.setter
    def image_mean(self, value):
        """
        Sets the image mean used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_mean = value
        self.config['image_mean'] = value
        self._update_normalize_tensor_transform()

    @property
    def image_std_dev(self):
        return self._image_std_dev

    @image_std_dev.setter
    def image_std_dev(self, value):
        """
        Sets the image std dev used in normalizing the images before
        being passed through the network
        :param value: list of floats
        :type value:
        :return:
        :rtype:
        """
        self._image_std_dev = value
        self.config['image_std_dev'] = value
        self._update_normalize_tensor_transform()

    @property
    def image_to_tensor(self):
        return self._image_to_tensor

    @image_to_tensor.setter
    def image_to_tensor(self, value):
        self._image_to_tensor = value

    @property
    def normalize_tensor_transform(self):
        return self._normalize_tensor_transform

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in [NetworkMode.TRAIN, NetworkMode.TEST]
        self._mode = value

    def _update_normalize_tensor_transform(self):
        """
        Updates the image to tensor transform using the current image mean and
        std dev
        :return: None
        :rtype:
        """
        self._normalize_tensor_transform = transforms.Normalize(self.image_mean, self.image_std_dev)

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

        return self.forward(img_tensor)


    def forward_on_img_tensor(self, img):
        """
        Deprecated, use `forward` instead
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

    def forward(self, img_tensor):
        """
        Simple forward pass on the network.
        Normalize the image if we are in TEST mode
        If we are in TRAIN mode then assume the dataset object has already normalized
        the image
        :param img_tensor: input tensor img.shape = [N,descriptor_dim, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: same as input type
        :rtype:
        """

        # Normalize the image if we are in TEST mode
        if self.mode == NetworkMode.TEST:
            img_tensor = self.normalize_tensor_transform(img_tensor)

        return self.fcn(img_tensor)

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

    def clip_pixel_to_image_size_and_round(self, uv):
        """
        Clips pixel to image coordinates and converts to int
        :param uv:
        :type uv:
        :return:
        :rtype:
        """
        u = min(int(round(uv[0])), self._image_width - 1)
        v = min(int(round(uv[1])), self._image_height - 1)
        return [u, v]

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



        dcn =  DenseCorrespondenceNetwork(fcn, config['descriptor_dimension'],
                                          image_width=config['image_width'],
                                          image_height=config['image_height'])

        dcn.config = config
        return dcn

    @staticmethod
    def find_best_match(pixel_a, res_a, res_b, debug=False):
        """
        Compute the correspondences between the pixel_a location in image_a
        and image_b

        :param pixel_a: vector of (u,v) pixel coordinates
        :param res_a: array of dense descriptors res_a.shape = [W,H,D]
        :param res_b: array of dense descriptors
        :param pixel_b: Ground truth . . .
        :return: (best_match_uv, best_match_diff, norm_diffs)
        best_match_idx is again in (u,v) = (right, down) coordinates

        """


        descriptor_at_pixel = res_a[pixel_a[1], pixel_a[0]]
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

        norm_diffs = np.sqrt(np.sum(np.square(res_b - descriptor_at_pixel), axis=2))

        best_match_flattened_idx = np.argmin(norm_diffs)
        best_match_xy = np.unravel_index(best_match_flattened_idx, norm_diffs.shape)
        best_match_diff = norm_diffs[best_match_xy]

        best_match_uv = (best_match_xy[1], best_match_xy[0])

        return best_match_uv, best_match_diff, norm_diffs

    def evaluate_descriptor_at_keypoints(self, img, keypoint_list, res=None):
        """

        :param dcn:
        :type dcn: DenseCorrespondenceNetwork
        :param img:
        :type img: numpy array (RGB)
        :param kp: list of cv2.KeyPoint
        :type kp:
        :param res: optionally pass in result of forward pass on network
        :return: numpy.ndarray (N,D) N = num keypoints, D = descriptor dimension
        This is the same format as sift.compute from OpenCV
        :rtype:
        """

        if res is None:
            res = self.forward_on_img(img)

        N = len(keypoint_list)
        D = self.descriptor_dimension
        des = np.zeros([N,D])

        for idx, kp in enumerate(keypoint_list):
            uv = self.clip_pixel_to_image_size_and_round([kp.pt[0], kp.pt[1]])
            des[idx,:] = res[uv[1], uv[0], :]

        # cast to float32, need this in order to use cv2.BFMatcher() with bf.knnMatch
        des = np.array(des, dtype=np.float32)
        return des

