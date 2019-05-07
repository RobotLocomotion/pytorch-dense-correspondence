from __future__ import print_function
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np


from torchvision import transforms as T
import torch
import cv2

from maskrcnn_benchmark.config import cfg


import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence_manipulation.maskrcnn_tools.predictor import COCODemo

"""
Wrapper for MaskRCNN predictions
"""

class MaskRCNNInference(object):

    def __init__(self, coco_demo):
        self._coco_demo = coco_demo


    @property
    def coco_demo(self):
        return self._coco_demo


    @staticmethod
    def save_predictions(save_dir, predictions):
        """

        :param save_dir:
        :type save_dir:
        :param predictions: maskrcnn_benchmark.structures.bounding_box.BoxList
        :type predictions:
        :return:
        :rtype:
        """

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # num detections
        bbox = predictions.bbox # torch.Tensor Nx4
        N = bbox.shape[0]
        masks = predictions.get_field("mask") # N x 1 x H x W


        d = dict()
        d['mode'] = predictions.mode # xyxy typically
        d['detections'] = dict()

        for i in range(0,N):
            data = dict()
            data['bbox'] = bbox[i,:].numpy().tolist()

            mask_image_filename = "{:d}_mask.png".format(i)
            data['mask_image_filename'] = mask_image_filename

            mask_PIL = Image.fromarray(masks[i, :, :, :].squeeze(0).numpy())
            mask_PIL.save(os.path.join(save_dir, mask_image_filename))

            d['detections'][i] = data


        pdc_utils.saveToYaml(d, os.path.join(save_dir, "data.yaml"))


    @staticmethod
    def from_model_folder(path_to_folder, config_file=None, weights_file=None):
        """
        Constructs a model from the specified folder.
        :param path_to_folder: absolute path to folder
        :type path_to_folder:
        :return:
        :rtype:
        """

        if config_file is None:
            yaml_files = glob.glob(path_to_folder + "/*.yaml")
            if len(yaml_files) > 1:
                raise ValueError("more than one file with .yaml extension, you must"
                                 "manually specify the config file")

            if len(yaml_files) == 0:
                raise ValueError("no yaml file found, is the config file in the model"
                                 "directory?")

            config_file = yaml_files[0]


        if weights_file is None:
            weights_file = os.path.join(path_to_folder, "model_final.pth")
            if not os.path.exists(weights_file):
                raise ValueError("model_final.pth doesn't exist, you must manually"
                                 "specify the weights_file")


        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        cfg.MODEL.WEIGHT = weights_file


        print("\n\nLoading MaskRCNN model")
        print("config file: %s" %(config_file))
        print("weights file: %s" %(weights_file))
        print("\n\n")

        coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

        return MaskRCNNInference(coco_demo)




