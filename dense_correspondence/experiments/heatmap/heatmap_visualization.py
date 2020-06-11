import os

# torch
import torch

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename, set_cuda_visible_devices
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader

# dense correspondence manipulation
from dense_correspondence_manipulation.visualization.heatmap_visualization import HeatmapVisualization
from dense_correspondence_manipulation.utils import dev_utils

set_cuda_visible_devices([1])



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
multi_episode_dict = dev_utils.load_flip_right_boot_episodes()
# model_file = dev_utils.get_real_right_shoes_dynamic_model_file()
model_file = dev_utils.get_flip_boot_dynamic_model_file()
camera_list = ["d415_02"]


######### REAL RIGHT MANY SHOES #############
# multi_episode_dict = dev_utils.load_shoe_imitation_episodes()
# model_file = dev_utils.get_real_right_shoes_dynamic_model_file()
# camera_list = ["d415_02"]


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



