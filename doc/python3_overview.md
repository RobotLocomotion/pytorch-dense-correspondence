# Python3 and Pytorch 1.2 update
This document outlines the basic structure of how to load a dataset and train a model.



## Dataset Class
A dataset fundamentally consists of pairs of RGBD images that are registered. By registered we mean that we know th
transform between the two images. The smallest unit of a dataset is what we call an **Episode**.

### Episode
An episode is a sub-unit f a dataset. So far we support two types of episodes.
- **static scene moving camera**: This is the type of data used in the original DenseObjectNets paper.
- **dynamic scene, multiple cameras**: This is the type of data used in the SSCVPL paper.

We support using your own data storage format. All that is required is that you be able to return
- RGBD images
- Masks (optional, could just make it the whole image)
- `T_W_C`: camera to world transform.

The basic class that outlines this interface is specified by the 
[`EpisodeReader` ](../dense_correspondence/dataset/episode_reader.py)
class.
