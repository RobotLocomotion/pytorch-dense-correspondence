#! /usr/bin/env directorPython

import os
import torch


# pdc
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence_manipulation.mesh_processing.mesh_descriptors import MeshDescriptors




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
                                       'caterpillar_single_scene_test.yaml')
    dataset_config = utils.getDictFromYamlFilename(dataset_config_file)
    dataset = SpartanDataset(config=dataset_config)

    scene_name = "2018-04-10-16-02-59"

    mesh_descriptors = MeshDescriptors(scene_name, dataset, dcn)

    # mesh_descriptors.compute_cell_descriptors()
    mesh_descriptors.process_cell_descriptors()

    print "finished cleanly"