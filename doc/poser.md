# Poser API

## Introduction

The purpose of this document is to specify the API for using `poser` to do registration tasks including:

- affine transformation between a canonical model and an observation (with learned features for correspondence)


## Poser interface overview

Based on discussion so far here is proposed API.

At a high level the interface is:

- *input*: a `poser_request.yaml` file, which specifies paths to data
- *output*: a `poser_out.yaml` file containing the output computed by poser

The idea is to be able to call the poser executable (`poser_don_app`) with a single argument, for example:

```
/full/path/to/poser_don_app /full/path/to/some_folder/poser_request.yaml
```

And then `poser_don_app` would create the `poser_out.yaml` file in this location:

```
/full/path/to/some_folder/poser_out.yaml
```

## Data input format

The description of the input format in a `poser_request.yaml` is:

```
object_1_name:
  template: /path/to/model_1.pcd
  image_1:
    descriptor_img: /path/to/img.png
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    visualize: 1
    camera_to_world:
      quaternion:
        w: 0.0
        x: 0.0
        y: 0.0
        z: 1.0
      translation:
        x: 0.0
        y: 0.0
        z: 0.0
object_2_name:
  template: /path/to/model_2.pcd
  image_1:
    descriptor_img: /path/to/img.png
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    visualize: 1
    camera_to_world:
      quaternion:
        w: 0.0
        x: 0.0
        y: 0.0
        z: 1.0
      translation:
        x: 0.0
        y: 0.0
        z: 0.0
```


By example here is a proposed valid `poser_request.yaml` file:

```
shoe_1:
  template: /home/wei/shoe_model.pcd
  image_1:
    descriptor_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/descriptor_images/000001_depth.png
    rgb_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_rgb.png
    depth_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_depth.png
    mask_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/mask_rcnn/000001/mask_001.png
    visualize: 1
    camera_to_world:
      quaternion:
        w: 0.13912495440375003
        x: -0.6406526750419419
        y: 0.7400920613829872
        z: -0.14990709690217974
      translation:
        x: 0.30046569018614927
        y: 0.003105128632966618
        z: 0.8281180455529766
shoe_2:
  template: /home/wei/shoe_model.pcd
  image_1:
    descriptor_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/descriptor_images/000001_depth.png
    rgb_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_rgb.png
    depth_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_depth.png
    mask_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/mask_rcnn/000001/mask_002.png
    visualize: 1
    camera_to_world:
      quaternion:
        w: 0.13912495440375003
        x: -0.6406526750419419
        y: 0.7400920613829872
        z: -0.14990709690217974
      translation:
        x: 0.30046569018614927
        y: 0.003105128632966618
        z: 0.8281180455529766
```

Note from above:

- the data does not need to be copied anywhere, the .yaml can just specify their paths
- it is up to the user to decide where the mask_img comes from (could be from mask_rcnnk, or "ground truth" mask)
- The mask should be a image at the same resolution of RGBD and have one channel uint8. The value 0 will be interperted as background, and all other values are foreground.
- The visualization field is a 0-1 optional flag. If the flag is 1, the `poser_don_app` will create a window to visualize the registration result. The window is blocking and you need to close the window manually to continue. 

## Data output format

The output `poser_out.yaml` copies all of the input data (for redundancy and no ".yaml correspondence" problem) and also adds the computed result:

```
object_1_name:
  template: /path/to/model_1.pcd
  image_1:
    descriptor_img: /path/to/img.png
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    camera_to_world:
      quaternion:
        w: 0.0
        x: 0.0
        y: 0.0
        z: 1.0
      translation:
        x: 0.0
        y: 0.0
        z: 0.0
     affine_model_to_observation: # some 4 x 4 matrix
object_2_name:
  template: /path/to/model_2.pcd
  image_1:
    descriptor_img: /path/to/img.png
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    camera_to_world:
      quaternion:
        w: 0.0
        x: 0.0
        y: 0.0
        z: 1.0
      translation:
        x: 0.0
        y: 0.0
        z: 0.0
    affine_model_to_observation: # some 4 x 4 matrix
```

