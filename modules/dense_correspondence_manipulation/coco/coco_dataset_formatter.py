from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from pycocotools import mask
import datetime
from skimage import measure
import json
import os
import random
import imageio
import glob


from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType
import dense_correspondence_manipulation.utils.utils as utils

# from caffe2.python import workspace
#
# # add desired local paths to PYTHONPATH
# import setup_path
# from detectron.core.config import assert_and_infer_cfg
# from detectron.core.config import cfg
# from detectron.core.config import merge_cfg_from_file
# from detectron.utils.io import cache_url
# from detectron.utils.timer import Timer
# import detectron.core.test_engine as infer_engine
# import detectron.utils.c2 as c2_utils
#
# c2_utils.import_detectron_ops()


class COCOSpartanDataset(SpartanDataset):
    """
    Inherits from SpartanDataset and is used to get relevant data for producing
    COCO style dataset annotations.
    """

    def __init__(self, *args, **kwargs):
        super(COCOSpartanDataset, self).__init__(*args, **kwargs)

    def get_object_id_list(self):
        # for example, returns ["shoes", "caterpillar", ...]
        return self._single_object_scene_dict.keys()

    def get_scenes_list_from_object_id(self, object_id):
        return self._single_object_scene_dict[object_id][self.mode]

    def get_image_indices_from_scene(self, scene):
        return self.get_pose_data(scene).keys()

    def maskrcnn_data_generator(self):
        # returns data equivalent to self.get_rgbd_mask_pose
        # goes over all data and yield data until empty

        object_ids = self.get_object_id_list()
        for object_id in object_ids:
            print("object_id: {}".format(object_id))
            scenes = self.get_scenes_list_from_object_id(object_id)
            for scene_name in scenes:
                print("scene_name: {}".format(scene_name))
                image_indices = sorted(self.get_image_indices_from_scene(scene_name))
                for image_idx in image_indices:
                    image_rgb, image_depth, image_mask, image_pose = self.get_rgbd_mask_pose(scene_name, image_idx)

                    # get the filename
                    rgb_filename = self.get_image_filename(scene_name, image_idx, ImageType.RGB)

                    yield object_id, scene_name, image_rgb, image_depth, image_mask, image_pose, rgb_filename

        raise ValueError("No more data left.")

    def get_random_maskrcnn_data(self):
        """
        This method returns the data needed for creating a COCO dataset.

        returns object_id, scene_name, image_rgb, image_depth, image_mask, image_pose, rgb_filename
        """

        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)
        image_idx = self.get_random_image_index(scene_name)
        image_rgb, image_depth, image_mask, image_pose = self.get_rgbd_mask_pose(scene_name, image_idx)

        # get the filename
        rgb_filename = self.get_image_filename(scene_name, image_idx, ImageType.RGB)

        return object_id, scene_name, image_rgb, image_depth, image_mask, image_pose, rgb_filename


class COCOSpartanDatasetLoader(object):
    """
    This class pulls images and masks from pytorch-dense-correspondences data.
    This will make the correct calls to SpartanDataset for loading COCO data.
    """

    def __init__(self, config_filename, train_config_filename, random=False):

        # TODO(ethan): understand why both a config file and train_config file are needed
        config = utils.getDictFromYamlFilename(config_filename)

        self.dataset = COCOSpartanDataset(config=config)


        # initialize some values
        self.object_id = None
        self.scene_name = None
        self.image_rgb = None
        self.image_mask = None
        self.rgb_filename = None

        self.maskrcnn_data_generator = self.dataset.maskrcnn_data_generator()

        # grab random data
        self.random = random

    def update_data(self):
        """
        Updates the class with a new datapoint. Store the data in class variables
        :return:
        :rtype:
        """

        if not self.random:
            object_id, scene_name, image_rgb, image_depth, image_mask, image_pose, rgb_filename = next(
                self.maskrcnn_data_generator)
        else:
            object_id, scene_name, image_rgb, image_depth, image_mask, image_pose, rgb_filename = self.dataset.get_random_maskrcnn_data()


        self.object_id = object_id
        self.scene_name = scene_name
        self.image_rgb = np.asarray(image_rgb)
        self.image_mask = np.asarray(image_mask)
        self.rgb_filename = rgb_filename

    def get_image_mask_pair(self):
        """
        Return rgb img, mask, rgb_filename for current datapoint
        :return:
        :rtype:
        """


        # load with new data
        self.update_data()

        return self.image_rgb, self.image_mask, self.rgb_filename

    def get_scene_name(self):
        """
        Returns the scene name of the current data point
        :return:
        :rtype:
        """
        return self.scene_name

    def get_object_id(self):
        """
        Return object_id for current datapoint
        :return:
        :rtype:
        """
        return self.object_id


class COCODataFormatter(object):
    """
    This class is used for taking an input image and a mask (as png file)
    and converting it to the correct format for COCO annotations.

    COCO format described here: http://cocodataset.org/#format-data
    """

    INFO = {
        "description": "Dense Correspondence COCO Dataset",
        "url": "https://github.com/RobotLocomotion/pytorch-dense-correspondence",
        "version": "",
        "year": 2018,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]

    CATEGORIES = [
        {
            'id': 0,
            'name': 'shoe',  # TODO(ethan): figure out where this is used and dynamically change it
            'supercategory': 'objects',
        }
    ]

    def __init__(self, name_of_set, base_folder_path=None):
        """

        :param name_of_set:
        :type name_of_set:
        :param annotations_path: (optional) where to save annotations
        :type annotations_path:
        :param images_path: (optional) where to save images
        :type images_path:
        """

        self.image_rgb = None
        self.image_mask = None

        # also do rle encoding
        self.encoded_mask = None

        # this will be for name_of_set.json and custom/name_of_set/*.jpg images
        self.name_of_set = name_of_set

        # set the path for where data should be saved
        current_path = os.path.dirname(os.path.realpath(__file__))

        if base_folder_path is None:
            base_folder_path = COCODataFormatter.get_data_path()

        self.annotation_path = os.path.join(base_folder_path, self.name_of_set)

        self.images_path = os.path.join(base_folder_path, self.name_of_set, 'images')

        # create this path if needed
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

        if not os.path.exists(self.annotation_path):
            os.makedirs(self.annotation_path)

    @staticmethod
    def get_data_path():
        return os.getenv("COCO_CUSTOM_DATA_DIR")

    def get_contours(self, image_mask):
        """
        return contours for the mask image

        contours of type [ [list of points], [list of points], etc.] for each contour
        """

        # findContours modifies the image, so make a copy
        mask_new, contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def get_height_and_width_of_image_mask(self, image_mask):
        """
        Get height and width of mask image
        :param image_mask:
        :type image_mask:
        :return: tupe (height, width)
        :rtype:
        """


        height, width = image_mask.shape[:2]
        return (height, width)

    def get_encoded_image_mask(self, image_mask):
        """
        Encode binary mask using RLE.

        See https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        :param image_mask:
        :type image_mask:
        :return:
        :rtype:
        """
        return mask.encode(np.asfortranarray(image_mask))

    def get_image_info(self,
                       image_id,
                       width,
                       height,
                       file_name,
                       license_id=1,
                       flickr_url="",
                       coco_url="",
                       date_captured=datetime.datetime.utcnow().isoformat(' ')):
        """
        Returns image data in the correct format for COCO
        """

        image = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": license_id,
            "flickr_url": flickr_url,
            "coco_url": coco_url,
            "date_captured": date_captured,
        }

        return image

    def get_polygons(self, image_mask, tolerance=0):
        """
        code from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py

        Args:

            binary_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.
        """

        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(image_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        return polygons

    def get_annotation_info(self, annotation_id, image_id, category_id, image_mask):
        """
        referencing code from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py

        :param annotation_id:
        :type annotation_id:
        :param image_id:
        :type image_id:
        :param category_id:
        :type category_id:
        :param image_mask:
        :type image_mask:
        :return:
        :rtype:
        """

        # this was set in the initialize function
        encoded_mask = self.get_encoded_image_mask(image_mask)

        area = self.get_area_of_encoded_mask(encoded_mask)
        if area < 1:
            raise ValueError("The area of the mask was 0.")

        x, y, width, height = self.get_bounding_box(encoded_mask)

        # using polygon (iscrowd = 0)
        segmentation = self.get_polygons(image_mask)

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": float(area),
            "bbox": [x, y, width, height],
            "iscrowd": 0,  # assume polygon for now
        }

        return annotation

    def get_category(self):
        # categories[{
        #     "id": int,
        #     "name": str,
        #     "supercategory": str,
        # }]
        pass

    def get_area_of_image_mask(self, image_mask):
        # return the area of the mask (by counting the nonzero pixels)

        # 1s correspond to the object
        mask_area = np.count_nonzero(image_mask)
        return mask_area

    def get_area_of_encoded_mask(self, encoded_mask):
        # return the area of the mask (by counting the nonzero pixels)

        # note that this also works
        return mask.area(encoded_mask)

    def get_encoded_rle_format(self, image_mask):
        fortran_ground_truth_binary_mask = np.asfortranarray(image_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        return encoded_ground_truth

    def get_bounding_box(self, encoded_mask):
        # returns x, y (top left), width, height
        bounding_box = mask.toBbox(encoded_mask)
        return bounding_box.astype(int)

    def get_coco_output_from_images_and_annotations(self, images, annotations):
        # use images and annotations to create the coco output dictionary

        # construct the dictionary that will be used to create the coco json data
        coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": images,
            "annotations": annotations
        }

        return coco_output

    def write_to_relative_json_filename(self, coco_output):
        # write the coco data to a json file which was specified at the class instantiation
        # self.name_of_set should be the string name to write to (which shouldn't include
        # .json)

        full_json_filename = os.path.join(self.annotation_path, self.name_of_set)

        with open('{}.json'.format(full_json_filename), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    def merge_images(self, background_image, background_mask, foreground_image, foreground_mask,
                     foreground_mask_value=2):
        # foreground_mask_value sets the pixel values to the specified value
        # (which is later used to create the labels)
        # returns merged_image, merged_mask

        # create the new foreground image and mask
        three_channel_foreground_mask = np.zeros_like(foreground_image)
        for i in range(3):
            three_channel_foreground_mask[:, :, i] = foreground_mask
        new_foreground_image = foreground_image * three_channel_foreground_mask
        new_foreground_mask = foreground_mask * foreground_mask_value

        # create the new background image and mask
        temporary_three_channel_foreground_mask = three_channel_foreground_mask.copy()
        # this is needed because the mask is no longer just binary (instead it has values corresponding to different objects)
        temporary_three_channel_foreground_mask[temporary_three_channel_foreground_mask > 0] = 1
        three_channel_foreground_mask_complement = np.ones_like(
            three_channel_foreground_mask) - temporary_three_channel_foreground_mask
        new_background_image = background_image * three_channel_foreground_mask_complement
        new_background_mask = background_mask * three_channel_foreground_mask_complement[:, :, 1]

        merged_image = new_background_image + new_foreground_image
        merged_mask = new_background_mask + new_foreground_mask

        return merged_image, merged_mask

    def get_image_mask_pair(self, data_loader, save_image=True):
        """
        Get an image, optionally also save the image for coco style training
        :param data_loader:
        :type data_loader:
        :param save_image:
        :type save_image:
        :return:
        :rtype:
        """

        current_image_rgb, current_image_mask, _ = data_loader.get_image_mask_pair()

        # save the rgb image so that we can return the filename as well
        if save_image:
            rgb_filename = self.save_rgb_image(current_image_rgb)
        else:
            rgb_filename = None

        return current_image_rgb, current_image_mask, rgb_filename

    def get_random_merged_image(self, data_loader, min_items, max_items):
        # takes in the loader
        # returns merged_image, merged_mask, rgb_filename (where rgb_filename is the full path)

        current_image_rgb, current_image_mask, _ = data_loader.get_image_mask_pair()

        num_shoes_in_image = random.randint(min_items, max_items)
        for i in range(2, 1 + num_shoes_in_image):
            image_rgb, image_mask, _ = data_loader.get_image_mask_pair()
            current_image_rgb, current_image_mask = \
                self.merge_images(current_image_rgb, current_image_mask, image_rgb, image_mask, foreground_mask_value=i)

        # save the rgb image so that we can return the filename as well
        rgb_filename = self.save_rgb_image(current_image_rgb)

        return current_image_rgb, current_image_mask, rgb_filename

    def save_rgb_image(self, image_rgb):
        # TODO(ethan): need to save the generated images somewhere
        # returns the full path as a string to where the rgb image was saved
        # saves based on where self.images_path is

        images_glob_str = os.path.join(self.images_path, "*.png")
        image_str_list = sorted(glob.glob(images_glob_str))
        image_index = len(image_str_list)

        image_path = os.path.join(self.images_path, "{:05}.png".format(image_index))

        # save the image
        imageio.imwrite(image_path, image_rgb)

        # return the path of where the image was saved
        return image_path

