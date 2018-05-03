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

import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
from dense_correspondence.evaluation.plotting import normalize_descriptor
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork, NetworkMode

sys.path.append(os.path.join(os.path.dirname(__file__), "../simple-pixel-correspondence-labeler"))
from annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2

sd = SpartanDataset()

eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'evaluation.yaml')
eval_config = utils.getDictFromYamlFilename(eval_config_filename)
utils.set_cuda_visible_devices([0])
dce = DenseCorrespondenceEvaluation(eval_config)
dcn = dce.load_network_from_config("caterpillar_image_normalization")

annotated_data_yaml_filename = "/home/peteflo/code/data_volume/pdc/evaluation_labeled_data/caterpillar_cross_scene_labels.yaml"
annotated_data = utils.getDictFromYamlFilename(annotated_data_yaml_filename)

index_of_pair_to_display = 0

def draw_points(img, img_points_picked):
    for index, img_point in enumerate(img_points_picked):
        draw_reticle(img, int(img_point["u"]), int(img_point["v"]), label_colors[index])

# mouse callback function
def find_match_from_image1(event,x,y,flags,param):
    global img2_heatmap
    img2_heatmap = make_heatmap_for(res_a, res_b, x, y)
    
def find_match_from_image2(event,x,y,flags,param):
    global img1_heatmap
    img1_heatmap = make_heatmap_for(res_b, res_a, x, y)
    
def make_heatmap_for(first_res, second_res, x, y):
    best_match_uv, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match((x,y), first_res, second_res)
    return norm_diffs

def next_image_pair_from_saved():
    global img1, img2, index_of_pair_to_display, img1_descriptors, img2_descriptors
    global res_a, res_b, img1_heatmap, img2_heatmap
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

    img1_pil = sd.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx)
    img2_pil = sd.get_rgb_image_from_scene_name_and_idx(scene_name_2, image_2_idx)

    print img1_pil
    print img2_pil

    rgb_1_tensor = sd.rgb_image_to_tensor(img1_pil)
    rgb_2_tensor = sd.rgb_image_to_tensor(img2_pil)

    def norm_both(res_a, res_b):
        both_min = min(np.min(res_a), np.min(res_b))
        normed_res_a = res_a - both_min
        normed_res_b = res_b - both_min

        both_max = max(np.max(normed_res_a), np.max(normed_res_b))
        normed_res_a = normed_res_a / both_max
        normed_res_b = normed_res_b / both_max

        return normed_res_a, normed_res_b

    res_a = dcn.forward_single_image_tensor(rgb_1_tensor).data.cpu().numpy()
    res_b = dcn.forward_single_image_tensor(rgb_2_tensor).data.cpu().numpy()
    res_a, res_b = norm_both(res_a, res_b)

    img1_descriptors = numpy_to_cv2(res_a)
    img2_descriptors = numpy_to_cv2(res_b)

    img1 = pil_image_to_cv2(img1_pil)
    img2 = pil_image_to_cv2(img2_pil)

    draw_points(img1, img1_points_picked)
    draw_points(img2, img2_points_picked)

    img1_heatmap = make_heatmap_for(res_b, res_a, 0, 0)
    img2_heatmap = make_heatmap_for(res_a, res_b, 0, 0)

    index_of_pair_to_display += 1

next_image_pair_from_saved()

cv2.namedWindow('image1')
cv2.setMouseCallback('image1',find_match_from_image1)
cv2.namedWindow('image1_descriptors')
cv2.namedWindow('image1_heatmap')

cv2.namedWindow('image2')
cv2.setMouseCallback('image2',find_match_from_image2)
cv2.namedWindow('image2_descriptors')
cv2.namedWindow('image2_heatmap')

while(1):
    cv2.imshow('image1',img1)
    cv2.imshow('image2',img2)
    cv2.imshow('image1_descriptors',img1_descriptors)
    cv2.imshow('image2_descriptors',img2_descriptors)
    cv2.imshow('image1_heatmap',img1_heatmap)
    cv2.imshow('image2_heatmap',img2_heatmap)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('n'):
        print "HEY"
        next_image_pair_from_saved()
        
cv2.destroyAllWindows()