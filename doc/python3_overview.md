# Python3 and Pytorch 1.2 update
This document outlines the basic structure of how to load a dataset and train a model.



## Dataset
A dataset fundamentally consists of pairs of RGBD images that are registered. By registered we mean that we know th
transform between the two images. The smallest unit of a dataset is what we call an **Episode**.

### EpisodeReader
An episode is a sub-unit of a dataset. So far we support two types of episodes.
- **static scene moving camera**: This is the type of data used in the original DenseObjectNets paper.
- **dynamic scene, multiple cameras**: This is the type of data used in the SSCVPL paper.

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
It is useful to visualize the result of computing matches and non-matches on your dataset. You can see 
[this](https://github.com/RobotLocomotion/key_dynam/blob/lm-pdc-rebase-2/notebooks/pdc_drake_sim_dataset_test.ipynb) notebook for an example of how to do this.


## Training
Currently the training only supports using the heatmap loss. An example can be found in 
[`train_drake_sim_dynamic_heatmap.py`](../dense_correspondence/training/train_drake_sim_dynamic_heatmap.py). The heatmap
loss is detailed in the [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229) paper. In the future we should
also use the spatial expectation loss.

## Visualizing Results
See [this](../dense_correspondence/evaluation/visualize_learned_correspondences.ipynb) notebook for a simple example of visualizing learned correspondences. A more involved (and older) example can be found at.
