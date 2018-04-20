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

from annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config

sd = SpartanDataset()

annotated_data_yaml_filename = "/home/peteflo/code/modules/simple-pixel-correspondence-labeler/saved_annotated_data/complete_merged_data.yaml"
annotated_data = utils.getDictFromYamlFilename(annotated_data_yaml_filename)

index_of_pair_to_display = 0

def draw_points(img, img_points_picked):
    for index, img_point in enumerate(img_points_picked):
        draw_reticle(img, int(img_point["u"]), int(img_point["v"]), label_colors[index])

def next_image_pair_from_saved():
    global img1, img2, index_of_pair_to_display
    print annotated_data[index_of_pair_to_display]
    annotated_pair = annotated_data[index_of_pair_to_display]
    
    scene_name_1 = annotated_pair["image_a"]["scene_name"]
    scene_name_2 = annotated_pair["image_b"]["scene_name"] 

    image_1_idx = annotated_pair["image_a"]["image_idx"]
    image_2_idx = annotated_pair["image_b"]["image_idx"]

    img1_points_picked = annotated_pair["image_a"]["pixels"]
    img2_points_picked = annotated_pair["image_b"]["pixels"]

    print img1_points_picked
    print img2_points_picked

    img1 = pil_image_to_cv2(sd.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx))
    img2 = pil_image_to_cv2(sd.get_rgb_image_from_scene_name_and_idx(scene_name_2, image_2_idx))

    draw_points(img1, img1_points_picked)
    draw_points(img2, img2_points_picked)

    index_of_pair_to_display += 1


next_image_pair_from_saved()

cv2.namedWindow('image1')
cv2.namedWindow('image2')

while(1):
    cv2.imshow('image1',img1)
    cv2.imshow('image2',img2)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('n'):
        print "HEY"
        next_image_pair_from_saved()
        
cv2.destroyAllWindows()