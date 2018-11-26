import sys
import os
import cv2
import numpy as np
import copy

import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.image_utils as image_utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType


KEYPOINT_LIST = ["toe", "bottom_of_shoelaces",  "top_of_shoelaces", "heel"]
USE_FIRST_IMAGE = True # force using first image in each log
RANDOMIZE_TEST_TRAIN = False # randomize selecting
RANDOMIZE_SCENES = False
DRAWING_SCALE_CONFIG = 2.0

DATASET_CONFIG_FILENAME = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'dataset', 'composite', 'shoe_train_all_shoes.yaml')
COLOR_GREEN = np.array([0,255,0])

DEBUG = True


white = (255,255,255)
black = (0,0,0)

label_colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), (0,125,125), (125,125,0), (200,255,50), (255, 125, 220), (10, 125, 255)]

DRAWING_SCALE_CONFIG = 2.0
TEXT_COLOR = (0,0,0)

###

annotated_data = []

###

def draw_reticle(img, x, y, label_color):
    cv2.circle(img,(x,y),10,label_color,1)
    cv2.circle(img,(x,y),11,white,1)
    cv2.circle(img,(x,y),12,label_color,1)
    cv2.line(img,(x,y+1),(x,y+3),white,1)
    cv2.line(img,(x+1,y),(x+3,y),white,1)
    cv2.line(img,(x,y-1),(x,y-3),white,1)
    cv2.line(img,(x-1,y),(x-3,y),white,1)
        
def scale_image(img, scale):
    return cv2.resize(img, (0,0), fx=scale, fy=scale)

class KeypointAnnotationTool(object):

    def __init__(self, dataset,  keypoint_list):
        self._keypoint_list = keypoint_list
        self._scene_list = dataset.get_scene_list()
        self._num_keypoints = len(keypoint_list)
        self._setup_config()
        self._dataset = dataset
        self._clear_cache()
        self._setup_scene_list()

        if DEBUG:
            print "scene_list", self._scene_list

    def _setup_config(self):
        self._config = dict()
        self._config['use_first_image'] = USE_FIRST_IMAGE
        self._config['window_name'] = 'image'

    def _setup_scene_list(self):
        """
        Populates self._object_list and self._scene_list

        :return:
        :rtype:
        """
        self._object_list = self._dataset.get_list_of_objects()
        self._scene_dict = dict()

        for object_id in self._object_list:
            self._scene_dict[object_id] = self._dataset.get_scene_list_for_object(object_id)

        self._object_id_counter = None
        self._scene_counter = None

        self._scene_data_list = []
        for object_id in self._object_list:
            for scene_name in self._dataset.get_scene_list_for_object(object_id):
                self._scene_data_list.append([object_id, scene_name])

        self._scene_generator = self._get_scene_generator()

    def _get_scene_generator(self):
        """
        Generator for scenes
        :return:
        :rtype:
        """
        for idx, data in enumerate(self._scene_data_list):
            print "Scene %d of %d" %(idx, len(self._scene_data_list))
            yield data

    def _get_next_object_id_and_scene(self):
        if self._object_id_counter is None:
            self._object_id_counter = 0
            self._scene_counter = 0
        else:
            object_id = self._object_list[self._object_id_counter]
            self._scene_counter += 1
            if self._scene_counter >= len(self._scene_dict[object_id]):
                self._object_id_counter += 1
                if self._object_id_counter >= len(self._object_list):
                    return False, False, False
                self._scene_counter = 0

        return (True, self._object_list[self._object_id_counter], self._scene_counter)

    def _clear_cache(self):
        """
        Clears the the cache
        :return:
        :rtype:
        """
        self._cache = dict() # cache where information is stored
        self._cache['pick_points'] = dict() # dict with keypoint names as keys
        self._cache['current_keypoint_idx'] = 0

    def _reset_image(self):
        """
        Clears the annotations on the current image
        :return:
        :rtype:
        """
        self._cache['pick_points'] = dict()  # dict with keypoint names as keys
        self._cache['current_keypoint_idx'] = 0


        scene_name_a = self._cache['scene_name']
        image_a_idx = self._cache['image_idx']
        img_a = sd.get_rgb_image_from_scene_name_and_idx(scene_name_a, image_a_idx)

        img_a = image_utils.pil_image_to_cv2(img_a)
        img_a = scale_image(img_a, DRAWING_SCALE_CONFIG)

        self._cache['img'] = img_a
        self._cache['img_with_live_reticle'] = np.copy(img_a)
        self._update_window_text(self._cache['img'])


    def _get_new_image(self, object_id=None, scene_name=None):
        """
        Gets a new image, populates the cache with the relevant information
        :return:
        :rtype:
        """
        print "\n\n----------\n\n"
        self._clear_cache()
        sd = self._dataset


        if RANDOMIZE_SCENES:
            object_id = sd.get_random_object_id()
            scene_name_a = sd.get_random_single_object_scene_name(object_id)
        else:
            try:
                object_id, scene_name_a = self._scene_generator.next()
            except StopIteration:
                print "you have labeled all scenes!!!"
                img_w_text = np.copy(self._config['img'])
                text = "DONE WITH ALL SCENES!!!"
                cv2.putText(img_w_text, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0))
                cv2.imshow(self._config['window_name'], img_w_text)
                return



        ## NOT RANDOM
        # scene_name_a = "2018-05-14-22-10-53"

        if self._config['use_first_image']:
            image_a_idx = sd.get_first_image_index(scene_name_a)
        else:
            image_a_idx = sd.get_random_image_index(scene_name_a)

        img_a = sd.get_rgb_image_from_scene_name_and_idx(scene_name_a, image_a_idx)

        img_a = image_utils.pil_image_to_cv2(img_a)
        img_a = scale_image(img_a, DRAWING_SCALE_CONFIG)

        self._cache['img'] = img_a
        self._cache['img_with_live_reticle'] = np.copy(img_a)
        self._cache['object_id'] = object_id
        self._cache['scene_name'] = scene_name_a
        self._cache['image_idx'] = image_a_idx

        self._update_window_text(self._cache['img'])

        return [img_a, scene_name_a, image_a_idx, object_id]

    def _get_current_keypoint(self):
        """
        Returns the name of the current keypoint
        :return:
        :rtype:
        """
        return self._keypoint_list[self._cache['current_keypoint_idx']]

    def _skip_keypoint(self):
        current_keypoint = self._get_current_keypoint()
        print "skipping keypoint", current_keypoint
        self._cache['current_keypoint_idx'] += 1
        self._update_window_text(self._cache['img'])

    def _update_window_text(self, img):
        """
        Writes some text to the window
        Copies the image passed in, adds text and visualizes it
        :return:
        :rtype:
        """

        text = ""
        if self._cache['current_keypoint_idx'] < self._num_keypoints:
            current_keypoint = self._keypoint_list[self._cache['current_keypoint_idx']]
            text = current_keypoint
        else:
            text = "DONE, 's' to save"

        img_w_text = np.copy(img)
        cv2.putText(img_w_text, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0))
        cv2.imshow(self._config['window_name'], img_w_text)

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
            label_color = label_colors[self._cache['current_keypoint_idx'] % len(label_colors)]
            if self._cache['current_keypoint_idx'] >= self._num_keypoints:
                print "you have recorded all the keypoints, use 's' to save"
                self._update_window_text(self._cache['img'])
                return

            current_keypoint = self._keypoint_list[self._cache['current_keypoint_idx']]
            print "keypoint recorded:", current_keypoint

            self._cache['pick_points'][current_keypoint] = (x / DRAWING_SCALE_CONFIG, y / DRAWING_SCALE_CONFIG)
            draw_reticle(self._cache['img'], x, y, label_color)

            self._cache['current_keypoint_idx'] += 1
            self._update_window_text(self._cache['img'])
        else:
            img_w_reticle = np.copy(self._cache['img'])
            draw_reticle(img_w_reticle, x, y, COLOR_GREEN)
            self._update_window_text(img_w_reticle)
            # cv2.imshow(self._config['window_name'], img_w_reticle)



    def picked_points_to_savable_dict(self, picked_points):
        """
        Converts the picked points to a savable list
        :param picked_points:
        :type picked_points:
        :return:
        :rtype:
        """
        d = dict()
        for keypoint_name, u_v_tuple in picked_points.iteritems():
            u_v_dict = dict()
            u_v_dict["keypoint"] = keypoint_name
            u_v_dict["u"] = u_v_tuple[0]
            u_v_dict["v"] = u_v_tuple[1]
            d[keypoint_name] = u_v_dict
        return d


    def _make_savable_correspondence_pairs(self):
        """

        :return:
        :rtype:
        """

        new_dict = dict()

        new_dict["object_id"] = self._cache['object_id']
        new_dict["scene_name"] = self._cache['scene_name']
        new_dict["image_idx"] = self._cache['image_idx']
        new_dict["keypoints"] = self.picked_points_to_savable_dict(self._cache['pick_points'])

        return copy.copy(new_dict)


    def run(self):
        """
        Starts the cv2 loop
        :return:
        :rtype:
        """

        print "Using this KEYPOINT_LIST:", self._keypoint_list
        print "If you want to change this, edit KEYPOINT_LIST in this file"
        self._get_new_image()
        img = self._cache['img']

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)


        while (1):


            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'):
                self._skip_keypoint()
            elif k == ord('r'):
                print "resetting image"
                self._reset_image()
            elif k == ord('s'):
                if len(self._cache['pick_points']) != len(KEYPOINT_LIST):
                    print "Need exactly", len(KEYPOINT_LIST), " annotations!"
                    print "These should be:"
                    print KEYPOINT_LIST
                else:
                    print "saving"
                    new_dict = self._make_savable_correspondence_pairs()
                    annotated_data.append(new_dict)
                    utils.saveToYaml(annotated_data, "new_annotated_keypoints.yaml")
                    self._get_new_image()
            elif k == ord('n'):
                self._get_new_image()

        cv2.destroyAllWindows()



if __name__ == "__main__":
    config = utils.getDictFromYamlFilename(DATASET_CONFIG_FILENAME)
    sd = SpartanDataset(config=config)
    sd.set_train_mode()

    ka = KeypointAnnotationTool(sd, KEYPOINT_LIST)
    ka.run()