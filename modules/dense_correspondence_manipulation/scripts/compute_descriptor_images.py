"""
Compute descriptor images for a single scene
"""

import os
import torch
import numpy as np

# pdc
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.dataset.scene_structure import SceneStructure

"""
Computes descriptor images for a given scene and network. Saves them as 
npy files. 


Usage: Modify the global variables in CAPS as needed
"""


DC_SOURCE_DIR = utils.getDenseCorrespondenceSourceDir()
NETWORK_NAME = "caterpillar_M_background_0.500_3"
EVALUATION_CONFIG_FILENAME = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence',
                                   'evaluation', 'lucas_evaluation.yaml')
DATASET_CONFIG_FILE = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence', 'dataset', 'composite',
                                       'caterpillar_only_9.yaml')

SCENE_NAME = "2018-04-16-14-25-19"

SAVE_DIR = os.path.join("/home/manuelli/code/data_volume/pdc/logs_test",
                            SCENE_NAME, "processed",
                            "descriptor_images", NETWORK_NAME)


def compute_descriptor_images_for_single_scene(dataset, scene_name, dcn, save_dir):
    """
    Computes the descriptor images for a single scene
    :param dataset:
    :type dataset:
    :param scene_name:
    :type scene_name:
    :param dcn:
    :type dcn:
    :return:
    :rtype:
    """

    pose_data = dataset.get_pose_data(scene_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    counter = 1
    num_images = len(pose_data)
    for img_idx, data in pose_data.iteritems():
        rgb, depth, mask, pose = dataset.get_rgbd_mask_pose(scene_name, img_idx)

        # rgb is a PIL image
        rgb_tensor = dataset.rgb_image_to_tensor(rgb)
        res = dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()

        # save the file

        descriptor_filename = os.path.join(save_dir, SceneStructure.descriptor_image_filename(img_idx))
        np.save(descriptor_filename, res)

        print "descriptor_filename", descriptor_filename
        print "processing image %d of %d" %(counter, num_images)
        counter += 1



if __name__ == "__main__":
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_filename = os.path.join(dc_source_dir, 'config', 'dense_correspondence',
                                   'evaluation', 'lucas_evaluation.yaml')
    eval_config = utils.getDictFromYamlFilename(config_filename)
    default_config = utils.get_defaults_config()
    utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

    dce = DenseCorrespondenceEvaluation(eval_config)
    network_name = "caterpillar_M_background_0.500_3"
    dcn = dce.load_network_from_config(network_name)

    dataset_config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'dataset', 'composite',
                                       'caterpillar_only_9.yaml')
    dataset_config = utils.getDictFromYamlFilename(dataset_config_file)
    dataset = SpartanDataset(config=dataset_config)


    scene_name = SCENE_NAME
    save_dir = SAVE_DIR
    compute_descriptor_images_for_single_scene(dataset, scene_name, dcn, save_dir)


    print "finished cleanly"

