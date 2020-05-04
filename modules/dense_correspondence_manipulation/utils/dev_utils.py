import os


# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename, get_data_dir
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader

def load_dataset_config():
    # placeholder for now
    config_file = os.path.join(getDenseCorrespondenceSourceDir(),
                               'config/dense_correspondence/global/drake_sim_dynamic.yaml')
    config = getDictFromYamlFilename(config_file)

    return config

def load_integral_heatmap_3d_config():
    config_file = os.path.join(getDenseCorrespondenceSourceDir(),
                               'config/dense_correspondence/global/integral_heatmap_3d.yaml')
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


def load_flip_right_boot_episodes():
    episodes_root = os.path.join(os.getenv("DATA_ROOT"), "pdc/imitation/logs/real_right_many_shoes_se2")
    episode_list_config = getDictFromYamlFilename(os.path.join(getDenseCorrespondenceSourceDir(),
                                                               'config/dense_correspondence/dataset/dynamic/flip_real_right_boot.yaml'))

    multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(episode_list_config,
                                                                  episodes_root)

    return multi_episode_dict


def load_hat_imitation_episodes():
    pass



def get_push_box_model_file():
    model_file = os.path.join(get_data_dir(), "dev/experiments/heatmap_dynamic/trained_models/2020-02-27-00-32-58_3D_loss_resnet50__dataset_real_push_box/net_dy_epoch_3_iter_10000_model.pth")

    return model_file


def get_real_right_shoes_dynamic_model_file():
    model_file = os.path.join(get_data_dir(), "dev/experiments/heatmap_dynamic/trained_models/2020-03-11-15-15-50_3D_loss_resnet50__dataset_real_right_many_shoes_se2/net_dy_epoch_0_iter_25000_model.pth")

    return model_file