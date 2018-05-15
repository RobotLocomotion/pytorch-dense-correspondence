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
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

sys.path.append(os.path.join(os.path.dirname(__file__), "../simple-pixel-correspondence-labeler"))
from annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2


config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                               'dense_correspondence', 'evaluation', 'evaluation.yaml')
config = utils.getDictFromYamlFilename(config_filename)
default_config = utils.get_defaults_config()
utils.set_cuda_visible_devices([0])
dce = DenseCorrespondenceEvaluation(config)

network_name = "caterpillar_standard_params_3"
dcn = dce.load_network_from_config(network_name)
sd = dcn.load_training_dataset()

annotated_data_yaml_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), "data_volume/pdc/evaluation_labeled_data/caterpillar_cross_scene_labels.yaml")
annotated_data = utils.getDictFromYamlFilename(annotated_data_yaml_filename)

index_of_pair_to_display = 0

def draw_points(img, img_points_picked):
    for index, img_point in enumerate(img_points_picked):
        draw_reticle(img, int(img_point["u"]), int(img_point["v"]), label_colors[index])

# mouse callback function
def pick_point_from_image1(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        u = x
        v = y
        print u, v
        print res_a[v,u,:]
        print "is your descriptor"
        save_descriptor_to_yaml(res_a[v,u,:])

def save_descriptor_to_yaml(descriptor):
    """
    This function saves the descriptor to a yaml file.

    The descriptor is some D-dimensional vector, which was indexed into
    a H,W,D numpy array.

    So descriptor is a 1-dimensional numpy array of length D 
    """
    print descriptor.shape
    descriptor_dict = dict()
    descriptor_dict["descriptor"] = descriptor.tolist()
    descriptor_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), "../config", "new_descriptor_picked.yaml")
    utils.saveToYaml(descriptor_dict, descriptor_filename) 
    return
      
def make_heatmap_for(first_res, second_res, x, y):
    best_match_uv, best_match_diff, norm_diffs = DenseCorrespondenceNetwork.find_best_match((x,y), first_res, second_res)
    return norm_diffs

def next_image_from_datset():
    global img1, img2, index_of_pair_to_display, img1_descriptors, img2_descriptors
    global res_a, res_b, img1_heatmap, img2_heatmap
    print annotated_data[index_of_pair_to_display]
    annotated_pair = annotated_data[index_of_pair_to_display]
    
    object_id    = sd.get_random_object_id()
    scene_name_1 = sd.get_random_single_object_scene_name(object_id)

    image_1_idx = sd.get_random_image_index(scene_name_1)

    img1_pil = sd.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx)

    print img1_pil

    rgb_1_tensor = sd.rgb_image_to_tensor(img1_pil)

    res_a = dcn.forward_single_image_tensor(rgb_1_tensor).data.cpu().numpy()

    img1_descriptors = numpy_to_cv2(res_a)

    img1 = pil_image_to_cv2(img1_pil)


next_image_from_datset()

cv2.namedWindow('image1')
cv2.moveWindow('image1', 0, 0)
cv2.moveWindow('image1', 200, 200)
cv2.setMouseCallback('image1', pick_point_from_image1)
cv2.namedWindow('image1_descriptors')
cv2.moveWindow('image1_descriptors', 0, 0)
cv2.moveWindow('image1_descriptors', 200+640, 200)


while(1):
    cv2.imshow('image1',img1)
    cv2.imshow('image1_descriptors',img1_descriptors)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('n'):
        print "HEY"
        next_image_from_datset()
        
cv2.destroyAllWindows()