import os


# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader

def load_dataset_config():
    # placeholder for now
    config_file = os.path.join(getDenseCorrespondenceSourceDir(),
                               'config/dense_correspondence/global/drake_sim_dynamic.yaml')
    config = getDictFromYamlFilename(config_file)

    return config

def load_push_box_episodes():
    episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/imitation/logs/real_push_box")
    episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
                                                               'config/dense_correspondence/dataset/dynamic/real_push_box.yaml'))
    multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(episode_list_config,
                                                                  episodes_root)

    return multi_episode_dict


def load_caterpillar_episodes():
    episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/logs_proto")
    episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
                                                               'config/dense_correspondence/dataset/single_object/caterpillar_9_episodes.yaml'))
    multi_episode_dict = SpartanEpisodeReader.load_dataset(episode_list_config,
                                                           episodes_root)

    return multi_episode_dict


def load_shoe_imitation_episodes():
    episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/imitation/logs/real_right_many_shoes_se2")
    episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
                                                               'config/dense_correspondence/dataset/dynamic/real_right_many_shoes_se2.yaml'))

    multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(episode_list_config,
                                                                  episodes_root)

    return multi_episode_dict


def get_push_box_model_file():
    data_root = os.getenv('DATA_ROOT')
    model_file = os.path.join(os.getenv('DATA_ROOT'), "pdc/dev/experiments/heatmap_dynamic/trained_models/2020-02-27-00-32-58_3D_loss_resnet50__dataset_real_push_box/net_dy_epoch_3_iter_10000_model.pth")

    return model_file