# Simple Pixel Correspondence Labeler

## How to use (in this repo)


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

