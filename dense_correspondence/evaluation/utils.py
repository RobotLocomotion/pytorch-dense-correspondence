#!/usr/bin/python


import os
import dense_correspondence_manipulation.utils.utils as utils
import logging
utils.add_dense_correspondence_to_python_path()
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import scipy.stats as ss
import itertools

class PandaDataFrameWrapper(object):
    """
    A simple wrapper for a PandaSeries that protects from read/write errors
    """

    def __init__(self, columns):
        data = [np.nan] * len(columns)
        self._columns = columns
        self._df = pd.DataFrame(data=[data], columns=columns)

    def set_value(self, key, value):
        if key not in self._columns:
            raise KeyError("%s is not in the index" %(key))

        self._df[key] = value

    def get_value(self, key):
        return self._df[key]

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, value):
        self._series = value



class KeypointAnnotationsPandasTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
               'image_idx',
               'keypoint_name',
               'object_id',
               'u',
               'v'
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, KeypointAnnotationsPandasTemplate.columns)


    @staticmethod
    def convert_keypoint_annotations_file_to_dataframe(keypoint_annotations):
        """

        :param keypoint_annotations: list of dicts of keypoint annotations in format
        specified in README. i.e. one dict looks like

        - image_idx: 3705
              keypoints:
                bottom_of_shoelaces:
                  keypoint: bottom_of_shoelaces
                  u: 396.5
                  v: 308.0
                heel:
                  keypoint: heel
                  u: 387.5
                  v: 107.0
                toe:
                  keypoint: toe
                  u: 406.0
                  v: 373.5
                top_of_shoelaces:
                  keypoint: top_of_shoelaces
                  u: 391.5
                  v: 238.5
              object_id: shoe_red_nike
              scene_name: 2018-11-16-21-00-00

        :type keypoint_annotations:
        :return: pandas DataFrame
        :rtype:
        """

        pd_dataframe_list = []

        for d in keypoint_annotations:
            for keypoint, keypoint_data in d['keypoints'].iteritems():
                dft = KeypointAnnotationsPandasTemplate()
                dft.set_value('scene_name', d['scene_name'])
                dft.set_value('image_idx', d['image_idx'])
                dft.set_value('object_id', d['object_id'])
                dft.set_value('keypoint_name', keypoint_data['keypoint'])
                dft.set_value('u', keypoint_data['u'])
                dft.set_value('v', keypoint_data['v'])

                pd_dataframe_list.append(dft.dataframe)


        df = pd.concat(pd_dataframe_list)

        return df



