import os

# torch
import torch

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader

# dense correspondence manipulation
from dense_correspondence_manipulation.visualization.heatmap_visualization import HeatmapVisualization
from dense_correspondence_manipulation.utils import dev_utils

# DATA_ROOT = os.path.join(os.getenv("DATA_ROOT"), "pdc")
#
# def load_dataset_config():
#     # placeholder for now
#     config_file = os.path.join(getDenseCorrespondenceSourceDir(),
#                                'config/dense_correspondence/global/drake_sim_dynamic.yaml')
#     config = getDictFromYamlFilename(config_file)
#
#     return config
#
#
# def load_push_box_episodes():
#     episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/imitation/logs/real_push_box")
#     episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
#                                                                'config/dense_correspondence/dataset/dynamic/real_push_box.yaml'))
#     multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(episode_list_config,
#                                                                   episodes_root)
#
#     return multi_episode_dict
#
#
# def load_caterpillar_episodes():
#     episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/logs_proto")
#     episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
#                                                                'config/dense_correspondence/dataset/single_object/caterpillar_9_episodes.yaml'))
#     multi_episode_dict = SpartanEpisodeReader.load_dataset(episode_list_config,
#                                                            episodes_root)
#
#     return multi_episode_dict
#
#
# def get_push_box_model_file():
#     model_file = os.path.join(DATA_ROOT, "dev/experiments/heatmap_dynamic/trained_models/2020-02-27-00-32-58_3D_loss_resnet50__dataset_real_push_box/net_dy_epoch_3_iter_10000_model.pth")
#
#     return model_file
#
#
# def get_caterpillar_model_file():
#     raise NotImplementedError


config = dev_utils.load_dataset_config()

############ PUSH BOX REAL ###################
# multi_episode_dict = dev_utils.load_push_box_episodes()
# model_file = dev_utils.get_push_box_model_file()
# camera_list = None

############ CATERPILLAR_9 ############
# multi_episode_dict = dev_utils.load_caterpillar_episodes()
# model_file = dev_utils.get_caterpillar_model_file()
# camera_list = None

######### REAL RIGHT BOOT #############
# multi_episode_dict = dev_utils.load_flip_right_boot_episodes()
# model_file = dev_utils.get_real_right_shoes_dynamic_model_file()
# camera_list = ["d415_02"]


######### REAL RIGHT MANY SHOES #############
multi_episode_dict = dev_utils.load_shoe_imitation_episodes()
model_file = dev_utils.get_real_right_shoes_dynamic_model_file()
camera_list = ["d415_02"]


# should really use validation data, but will use train for now . . .
# will be cross-scene so that shouldn't matter . . . .
dataset = DynamicDrakeSimDataset(config, multi_episode_dict, phase="train")

# load model
model_config_file = os.path.join(os.path.dirname(model_file), 'config.yaml')
model_config = getDictFromYamlFilename(model_config_file)
#
# print("model_config", model_config)
# print("test", model_config['loss_function']['heatmap']['sigma_fraction'])


model = torch.load(model_file)
model = model.cuda()
model = model.eval()


heatmap_vis = HeatmapVisualization(model_config,
                                   dataset,
                                   model,
                                   visualize_3D=False,
                                   camera_list=camera_list)
heatmap_vis.run()



