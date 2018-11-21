# Simple Pixel Correspondence Labeler


## Regular Cross Scene Data

### Getting started
1. Be inside docker container for `pytorch-dense-correspondence`
2. `use_pytorch_dense_correspondence`
3. `python annotate_correspondences.py`

Two windows should pop up.  You can move them around.  Will explain more later in this readme...

### Changing dataset configuration

If you'd like to change dataset configuration, it's just a `SpartanDataset` object created at the start of `annotate_correspondences.py`,
and configs could be passed to this object.

### Labeling data

**Controls**

| Button        | Action        | 
| ------------- |:-------------:|
| left-click      | label point |
| mouse wheel      | zoom in/out      |
| n | go to a new random pair of images (does not save anything on current screen)   |
| s | add the current set of labels to the saved data (writes to disk, nothing else needed) | 

**Process**

Find a point in one of the images, go click on it, then find a point in the other image, and click on that.  *Recommend trying to find
5-6 correspondence points per image pair*.

The colors will sequentially help you line up which corresponds to which.

If you like the labels you've clicked on, save with "s", then use "n" to go to next random image pair.

If you messed up your labeling, just use "n" to go the next random image pair, it won't save anything.

**Note**

The icon of your mouse before you click is a "hand". It's approximately the "center of the palm" of the hand that defines exactly
the click point, not a finger or the top.

Easiest to super zoom in before clicking on a pixel.

### Re-visualizing your labeled correspondences

Want to look back at the correspondences you just labeled?

`python visualize_correspondences.py`

### Managing saved data

The saved data will be saved out as `new_annotated_pairs.yaml`.  For internal team use we want to put this somewhere in `pdc/evaluation_labeled_data/` and then add a pointer to this file in the `dataset.yaml`.

### Merging saved data

If you save out two different sessions of annotated data and want to merge them, `cat` has us covered:

```
cat new_annotated_pairs.yaml >> old_annotated_pairs.yaml
```

Which will append the new pairs onto the end of the old pairs yaml.

## Class Consistent Keypoint Annotations

### Labeling data
- Use the `modules/dense_correspondence_manipulation/simple_pixel_correspondence_labeler/annotate_keypoints.py` tool 
- Edit the top of the file to change `KEYPOINT_LIST = ["toe", "top_of_shoelaces", "heel"]` to whatever you want
- Edit the dataset configuration in the file too, any other parameters you see
- `python annotate_keypoints.py`
- One image at a time, label the keypoints in order as you specified
- When done with an image, press `s` to save and move on to the next image

Not yet supported:
- Currently all the keypoints need to be visible in a given image.  Could add support for some keypoints not being visible.

### Example Labeled Data

The resulting annotations are stored in `new_annotated_keypoints.yaml`. Example data is shown below for completeness.

```
- image:
    image_idx: 0
    object_id: shoe_red_nike
    pixels:
    - keypoint: top_of_shoelaces
      u: 279.0
      v: 281.5
    - keypoint: bottom_of_shoelaces
      u: 263.0
      v: 335.5
    - keypoint: heel
      u: 302.0
      v: 210.5
    - keypoint: toe
      u: 249.5
      v: 378.5
    scene_name: 2018-05-14-22-17-00

```

### Analyzing class-consistent keypoints

- Use the `dense_correspondence/evaluation/evaluation_quantitative_cross_scene.ipynb` tool
- Specify dataset configuration and path to labeled cross instance data
- Tools for displaying the plots are at the bottom of this file


