#! /usr/bin/env directorPython

import os
import torch

# pdc
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence_manipulation.mesh_processing.mesh_descriptors import MeshDescriptors

"""
Computes the descriptors for each cell in the mesh.

Usage: edit the variables in CAPS as needed.

COMPUTE_CELL_DESCRIPTORS:
    - this 

Then run compute_mesh_descriptors.py
"""

DC_SOURCE_DIR = utils.getDenseCorrespondenceSourceDir()
NETWORK_NAME = "caterpillar_M_background_0.500_3"
EVALUATION_CONFIG_FILENAME = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence',
                                   'evaluation', 'lucas_evaluation.yaml')
DATASET_CONFIG_FILE = os.path.join(DC_SOURCE_DIR, 'config', 'dense_correspondence', 'dataset', 'composite',
                                       'caterpillar_single_scene_test.yaml')

SCENE_NAME = "caterpillar"

COMPUTE_MESH_DESCRIPTORS = True
PROCESS_MESH_DESCRIPTORS = True

if __name__ == "__main__":
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    eval_config = utils.getDictFromYamlFilename(EVALUATION_CONFIG_FILENAME)
    default_config = utils.get_defaults_config()
    utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

    dce = DenseCorrespondenceEvaluation(eval_config)
    network_name = "caterpillar_M_background_0.500_3"
    dcn = dce.load_network_from_config(network_name)


    dataset_config = utils.getDictFromYamlFilename(DATASET_CONFIG_FILE)
    dataset = SpartanDataset(config=dataset_config)

    mesh_descriptors = MeshDescriptors(SCENE_NAME, dataset, dcn, NETWORK_NAME)


    if COMPUTE_MESH_DESCRIPTORS:
        print "\n-------computing cell descriptors---------\n"
        mesh_descriptors.compute_cell_descriptors()
        print "\n-------finished computing cell descriptors---------\n"

    if PROCESS_MESH_DESCRIPTORS:
        print "\n---------processing cell descriptors---------\n"
        mesh_descriptors.process_cell_descriptors()
        print "\n------- finished processing cell descriptors---------\n"

    print "finished cleanly"