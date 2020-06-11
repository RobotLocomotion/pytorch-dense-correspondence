import yaml
import os

def get_source_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_episode_list(composite_config):
    episode_list = set()

    for single_object_config_file in composite_config['single_object_scenes_config_files']:
        print("single_object_config_file", single_object_config_file)
        config_file = os.path.join(get_source_dir(),
                                   'config/dense_correspondence/dataset/single_object',
                                   single_object_config_file)
        config = yaml.safe_load(open(config_file, 'r'))

        episode_list.update(set(config['train']))
        episode_list.update(set(config['test']))


    for multi_object_config_file in composite_config['multi_object_scenes_config_files']:
        config_file = os.path.join(get_source_dir(),
                                   'config/dense_correspondence/dataset/multi_object',
                                   multi_object_config_file)
        config = yaml.safe_load(open(config_file, 'r'))

        episode_list.update(set(config['train']))
        episode_list.update(set(config['test']))



    episode_list = list(episode_list)

    config = {'episodes': episode_list}
    filename = "episodes.yaml"
    print("filename", filename)
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":
    # composite_config_file = "/home/manuelli/code/key_dynam_workspace/pdc/config/dense_correspondence/dataset/composite/shoe_train_all_shoes.yaml"

    composite_config_file = "/home/manuelli/code/key_dynam_workspace/pdc/config/dense_correspondence/dataset/composite/mugs_all.yaml"

    composite_config = config = yaml.safe_load(open(composite_config_file, 'r'))

    print(composite_config_file)
    print(composite_config)

    make_episode_list(composite_config)