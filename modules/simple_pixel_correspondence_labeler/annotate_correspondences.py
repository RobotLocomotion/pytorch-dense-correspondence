import sys
import os
import cv2
import numpy as np
import copy

import dense_correspondence_manipulation.utils.utils as utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'star_bot_front_only.yaml')
config = utils.getDictFromYamlFilename(config_filename)
sd = SpartanDataset(config=config)
sd.set_train_mode()


USE_FIRST_IMAGE = False # force using first image in each log
RANDOMIZE_TEST_TRAIN = False # randomize seletcting

def numpy_to_cv2(numpy_img):
    return numpy_img[:, :, ::-1].copy() # open and convert between BGR and RGB

def pil_image_to_cv2(pil_image):
    return np.array(pil_image)[:, :, ::-1].copy() # open and convert between BGR and RGB
def get_cv2_img_pair_from_spartan():
    scene_name_a = sd.get_random_scene_name()
    num_attempts = 50
    for i in range(num_attempts):
        if (i % 2) == 0 and RANDOMIZE_TEST_TRAIN:
            sd.set_train_mode()
        else:
            sd.set_test_mode()

        scene_name_b = sd.get_random_scene_name()
        if scene_name_b != scene_name_a:
            break    
        if i == (num_attempts - 1):
            print "Failed at randomly getting two different scenes"
            exit()
    
    if USE_FIRST_IMAGE:
        image_a_idx = 0
        image_b_idx = 0
    else:
        image_a_idx = sd.get_random_image_index(scene_name_a)
        image_b_idx = sd.get_random_image_index(scene_name_b)

    img_a = sd.get_rgb_image_from_scene_name_and_idx(scene_name_a, image_a_idx)
    img_b = sd.get_rgb_image_from_scene_name_and_idx(scene_name_b, image_b_idx)
    img_a, img_b = pil_image_to_cv2(img_a), pil_image_to_cv2(img_b)
    img_a, img_b = scale_image(img_a, drawing_scale_config), scale_image(img_b, drawing_scale_config)  
    return [img_a, scene_name_a, image_a_idx], [img_b, scene_name_b, image_b_idx]

####

white = (255,255,255)
black = (0,0,0)

label_colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), (0,125,125), (125,125,0), (200,255,50), (255, 125, 220), (10, 125, 255)]

drawing_scale_config = 2.0

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

# mouse callback function
def draw_circle1(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img1_points_picked
        this_pair_label_index = len(img1_points_picked)
        label_color = label_colors[this_pair_label_index%len(label_colors)]
        img1_points_picked.append((x/drawing_scale_config,y/drawing_scale_config))
        draw_reticle(img1, x, y, label_color)
        
def draw_circle2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img2_points_picked
        this_pair_label_index = len(img2_points_picked)
        label_color = label_colors[this_pair_label_index%len(label_colors)]
        img2_points_picked.append((x/drawing_scale_config,y/drawing_scale_config))
        draw_reticle(img2, x, y, label_color)
        
def check_same_length(img1_points_picked, img2_points_picked):
    return (len(img1_points_picked) == len(img2_points_picked))

def scale_image(img, scale):
    return cv2.resize(img, (0,0), fx=scale, fy=scale) 

def next_image_pair():
    global img1_points_picked, img2_points_picked, img1, scene_name_1, image_1_idx, img2, scene_name_2, image_2_idx
    img1_points_picked = []
    img2_points_picked = []
    [img1, scene_name_1, image_1_idx], [img2, scene_name_2, image_2_idx] = get_cv2_img_pair_from_spartan()

def to_savable_list(points_picked):
    savable_list = []
    for u_v_tuple in points_picked:
        u_v_dict = dict()
        u_v_dict["u"] = u_v_tuple[0]
        u_v_dict["v"] = u_v_tuple[1]
        savable_list.append(u_v_dict)
    return savable_list

def make_savable_correspondence_pairs():
    new_dict = dict()
    new_dict["image_a"] = dict()
    new_dict["image_b"] = dict()

    new_dict["image_a"]["scene_name"] = scene_name_1
    new_dict["image_b"]["scene_name"] = scene_name_2

    new_dict["image_a"]["image_idx"] = image_1_idx
    new_dict["image_b"]["image_idx"] = image_2_idx

    new_dict["image_a"]["pixels"] = to_savable_list(img1_points_picked)
    new_dict["image_b"]["pixels"] = to_savable_list(img2_points_picked)

    return copy.copy(new_dict)

if __name__ == "__main__":
    next_image_pair()

    cv2.namedWindow('image1')
    cv2.setMouseCallback('image1',draw_circle1)

    cv2.namedWindow('image2')
    cv2.setMouseCallback('image2',draw_circle2)

    while(1):
        cv2.imshow('image1',img1)
        cv2.imshow('image2',img2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print ix,iy
        elif k == ord('s'):
            if not check_same_length(img1_points_picked, img2_points_picked):
                print "can't save when not same length"
                print "try choosing a new image pair"
                print "or picking more points on the one with less points"
            else:
                print "saving"
                new_dict = make_savable_correspondence_pairs()
                annotated_data.append(new_dict)
                utils.saveToYaml(annotated_data, "new_annotated_pairs.yaml")
        elif k == ord('n'):
            next_image_pair()
            
    cv2.destroyAllWindows()
