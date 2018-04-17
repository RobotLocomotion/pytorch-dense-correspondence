import sys
import os
import cv2
import numpy as np

import dense_correspondence_manipulation.utils.utils as utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType

sd = SpartanDataset(debug=False)

def pil_image_to_cv2(pil_image):
    return np.array(pil_image)[:, :, ::-1].copy() # open and convert between BGR and RGB

def get_cv2_img_pair_from_spartan():
    scene_name_a = sd.get_random_scene_name()
    num_attempts = 50
    for i in range(num_attempts):
        scene_name_b = sd.get_random_scene_name()
        if scene_name_b != scene_name_a:
            break    
        if i == (num_attempts - 1):
            print "Failed at randomly getting two different scenes"
            exit()
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

label_colors = [(255,0,0), (0,255,0), (0,0,255)]

ix,iy = -1,-1

drawing_scale_config = 2.0

###

def draw_crosshairs(img, x, y):
    crosshair_length = 10
    cv2.line(img,(x,y+1),(x,y+10),white,1)
    cv2.line(img,(x+1,y),(x+10,y),white,1)
    cv2.line(img,(x,y-1),(x,y-10),white,1)
    cv2.line(img,(x-1,y),(x-10,y),white,1)

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
        supported_num_points = 3
        if this_pair_label_index >= supported_num_points:
            print "Currently only suggesting to do", supported_num_points, "points at a time"
            return
        label_color = label_colors[this_pair_label_index]
        img1_points_picked.append((x/drawing_scale_config,y/drawing_scale_config))
        draw_reticle(img1, x, y, label_color)
        
def draw_circle2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img2_points_picked
        this_pair_label_index = len(img2_points_picked)
        supported_num_points = 3
        if this_pair_label_index >= supported_num_points:
            print "Currently only suggesting to do", supported_num_points, "points at a time"
            return
        label_color = label_colors[this_pair_label_index]
        img2_points_picked.append((x/drawing_scale_config,y/drawing_scale_config))
        draw_reticle(img2, x, y, label_color)
        
def scale_image(img, scale):
    return cv2.resize(img, (0,0), fx=scale, fy=scale) 

def next_image_pair():
    global img1_points_picked, img2_points_picked, img1, scene_name_1, image_1_idx, img2, scene_name_2, image_2_idx
    img1_points_picked = []
    img2_points_picked = []
    [img1, scene_name_1, image_1_idx], [img2, scene_name_2, image_2_idx] = get_cv2_img_pair_from_spartan()

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
        print "saving"
        print scene_name_1
        print img1_points_picked
        print scene_name_2
        print img2_points_picked
    elif k == ord('n'):
        next_image_pair()
        
cv2.destroyAllWindows()
