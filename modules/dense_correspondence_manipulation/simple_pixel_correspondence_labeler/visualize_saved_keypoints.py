import sys
import os
import cv2
import numpy as np
import copy

import dense_correspondence_manipulation.utils.utils as utils
dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType

from dense_correspondence_manipulation.simple_pixel_correspondence_labeler.annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'caterpillar_baymax_starbot_onlymulti_front.yaml')
config = utils.getDictFromYamlFilename(config_filename)
sd = SpartanDataset(config=config)
sd.set_train_mode()

annotated_data_yaml_filename = os.path.join(os.getcwd(), "new_annotated_keypoints.yaml")
annotated_data = utils.getDictFromYamlFilename(annotated_data_yaml_filename)

index_of_pair_to_display = 0

def draw_points(img, img_points_picked):
    for index, img_point in enumerate(img_points_picked):
        color = label_colors[index%len(label_colors)]
        draw_reticle(img, int(img_point["u"]), int(img_point["v"]), color)

def next_image_from_saved():
    global img1, index_of_pair_to_display
    print annotated_data[index_of_pair_to_display]
    annotated_pair = annotated_data[index_of_pair_to_display]
    
    scene_name_1 = annotated_pair["image"]["scene_name"]
    
    image_1_idx = annotated_pair["image"]["image_idx"]

    img1_points_picked = annotated_pair["image"]["pixels"]

    print img1_points_picked

    img1 = pil_image_to_cv2(sd.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx))

    draw_points(img1, img1_points_picked)

    index_of_pair_to_display += 1


next_image_from_saved()

cv2.namedWindow('image1')


while(1):
    cv2.imshow('image1',img1)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('n'):
        next_image_from_saved()
        
cv2.destroyAllWindows()