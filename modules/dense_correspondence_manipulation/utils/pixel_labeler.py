import os
import cv2
import numpy as np

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.dataset.dynamic_spartan_dataset import DynamicSpartanDataset

from dense_correspondence_manipulation.simple_pixel_correspondence_labeler.annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2

SCENE_NAME = "2019-07-30-18-20-20"
CAMERA_NUM = 1
IMG_IDX = 150
LOGS_ROOT_PATH = "/home/manuelli/data_ssd/imitation/logs/push_box" # options are [str, None] overwrite if you need it to be different

DATASET_CONFIG = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'dynamic_sugar_push_box.yaml')
COLOR_GREEN = np.array([0,255,0])


class PixelLabeler(object):
    """
    Simple tools that allows you to annotate pixels in an image. Saves the results to a
    YAML file
    """

    def __init__(self):
        self._load_dataset()
        self._window_name = "Image"

    def _load_dataset(self):
        dataset_config = utils.getDictFromYamlFilename(DATASET_CONFIG)
        if LOGS_ROOT_PATH is not None:
            dataset_config["logs_root_path"] = LOGS_ROOT_PATH
        dataset = DynamicSpartanDataset(config=dataset_config)

        self._dataset = dataset

    def _load_image(self):
        img_PIL = self._dataset.get_rgb_image_from_scene_name_and_idx_and_cam(SCENE_NAME, IMG_IDX, CAMERA_NUM)

        self._cache['img'] = pil_image_to_cv2(img_PIL)
        self._cache['scene_name'] = SCENE_NAME
        self._cache['img_idx'] = IMG_IDX
        self._cache["camera_num"] = CAMERA_NUM

    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):
        """
        Draws reticle at the picked point
        :param event:
        :type event:
        :param x:
        :type x:
        :param y:
        :type y:
        :param flags:
        :type flags:
        :param param:
        :type param:
        :return:
        :rtype:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            label_color = label_colors[len(self._cache['xy_pixels']) % len(label_colors)]
            self._cache['xy_pixels'].append([x,y])
            draw_reticle(self._cache['img'], x, y, label_color)
            cv2.imshow(self._window_name, self._cache['img'])
        else:
            img_w_reticle = np.copy(self._cache['img'])
            draw_reticle(img_w_reticle, x, y, COLOR_GREEN)
            cv2.imshow(self._window_name, img_w_reticle)

    def run(self):
        self._cache = dict()
        self._cache['xy_pixels'] = []
        self._load_image()

        cv2.namedWindow(self._window_name)
        cv2.imshow(self._window_name, self._cache['img'])
        cv2.setMouseCallback(self._window_name, self.draw_circle)

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('s'):
                self.save()

    def save(self):
        """
        Save annotations to disk
        :return:
        :rtype:
        """
        filename = "clicked_points.yaml"
        print("saving data to", filename)
        save_data = {"xy_pixels": self._cache['xy_pixels'],
                     'camera_num': self._cache['camera_num'],
                     'img_idx': self._cache['img_idx'],
                     'scene_name': self._cache['scene_name']}

        utils.saveToYaml(save_data, filename)

if __name__ == "__main__":
    pl = PixelLabeler()
    pl.run()