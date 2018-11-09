# Labeled Data

This document outlines our tools and processes for human labelling data for network evaluation
purposes.

## Single Object Cross

## Class Consistent

### Labeling data
- Use the `modules/simple-pixel-correspondence-labeler/annotate_keypoints.py` tool 
- Edit the top of the file to change KEYPOINT_LIST = ["toe", "top_of_shoelaces", "heel"] to whatever you want
- Edit the dataset configuration in the file too, any other parameters you see
- `python annotate_keypoints.py`
- One image at a time, label the keypoints in order as you specified
- When done with an image, press `s` to save and move on to the next image

Not yet supported:
- Currently all the keypoints need to be visible in a given image.  Could add support for some keypoints not being visible.

### Analyzing class-consistent keypoints

- Use the `dense_correspondence/evaluation/evaluation_quantitative_cross_scene.ipynb` tool
- Specify dataset configuration and path to labeled cross instance data
- Tools for displaying the plots are at the bottom of this file
