# Python3 and Pytorch 1.2 update
This document outlines the basic structure of how to load a dataset and train a model.

## Environment Setup
My code setup for this project is

```
code/
  pdc/
  key_dynam/
```

where `pdc` is this repo and `key_dynam` is [this](https://github.com/RobotLocomotion/key_dynam/tree/lm-pdc-rebase-2) branch. I use the docker build in `key_dynam` which has the right dependencies and then have both codebases accessible inside there. You need to be careful to appropriately source things and put them on the `PYTHONPATH` so that you can import them.




## Dataset
A dataset fundamentally consists of pairs of RGBD images that are registered. By registered we mean that we know th
transform between the two images. The smallest unit of a dataset is what we call an **Episode**.

### EpisodeReader
An episode is a sub-unit of a dataset. So far we support two types of episodes.
- **static scene moving camera**: This is the type of data used in the original DenseObjectNets paper. 
[EpisodeReader](..//dense_correspondence/dataset/spartan_episode_reader.py), note this one is still a work in progress.
- **dynamic scene, multiple cameras**: This is the type of data used in the SSCVPL paper. [EpisodeReader](https://github.com/RobotLocomotion/key_dynam/blob/lm-pdc-rebase-2/dataset/drake_sim_episode_reader.py)

We support using your own data storage format. All that is required is that you be able to return

- RGBD images
- Masks (optional, could just make it the whole image)
- `T_W_C`: camera to world transform.

The basic class that outlines this interface is specified by the 
[`EpisodeReader` ](../dense_correspondence/dataset/episode_reader.py)
class.

### DatasetClass
The dataset class contains many EpisodeReader objects and samples from them. The current implementation
(which needs to be cleaned up) is [`DynamicDrakeSimDataset`](../dense_correspondence/dataset/dynamic_drake_sim_dataset.py). 
Currently the implementation is quite simple and doesn't include any data augmentation (but this should be added at a later 
date). The unit of data returned by the `__getitem__` method is a `dict` of the form

```
{'data_a': data_a,
'data_b': data_b,
'matches': matches_data,
'masked_non_matches': masked_non_matches_data,
'background_non_matches': background_non_matches_data,
'metadata': metadata,
'valid': True}
```

See the code for more detail on these output types. 

### Visualizing your dataset
It is useful to visualize the result of computing matches and non-matches on your dataset. Two examples of this are
- With real data from DenseObjectNets paper [here](../dense_correspondence/dataset/simple_dataset_test_episode_reader.ipynb)
- With simulated data from Drake. [here](https://github.com/RobotLocomotion/key_dynam/blob/lm-pdc-rebase-2/notebooks/pdc_drake_sim_dataset_test.ipynb).

### Creating dataset using drake simulator
This is implemented in the `key_dynam` repo. Check out [this](https://github.com/RobotLocomotion/key_dynam/blob/lm-pdc-rebase-2/experiments/05/collect_episodes.py) script.


## Training

We support two losses, **heatmap** and **spatial expectation** (both 2D and 3D), which are detailed in the [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229) paper. An example training script can be found in 
[`train_heatmap.py`](../dense_correspondence/training/train_drake_sim_dynamic_heatmap.py).

- For an example training on the caterpillar data from the original DON paper see [`train_heatmap.py`](../dense_correspondence/experiments/heatmap/train_heatmap.py)
- For an example of training with dynamic scene data from SSCVPL paper see [`train_dynamic_heatmap.py`](../dense_correspondence/experiments/heatmap/train_dynamic_heatmap.py)
- For an example of training with 3D loss see [`train_integral_heatmap_3d.py`](../dense_correspondence/experiments/heatmap/train_integral_heatmap_3d.py)



## Visualizing Results
- See [this](../dense_correspondence/evaluation/visualize_learned_correspondences.ipynb) notebook for a simple example of visualizing learned correspondences. 

- Interactive heatmap tool: See [`heatmap_visualization.py`](../dense_correspondence/experiments/heatmap/heatmap_visualization.py).
