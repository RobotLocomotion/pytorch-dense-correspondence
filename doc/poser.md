# Poser API

## Introduction

The purpose of this document is to specify the API for using `poser` to do registration tasks including:

- affine and rigid transformation between a canonical model and an observation (with learned features for correspondence)


## Poser interface overview

Based on discussion so far here is proposed API.

At a high level the interface is:

- *input*: a `poser_request.yaml` file, which specifies paths to data
- *output*: a `poser_out.yaml` file containing the output computed by poser

The idea is to be able to call the poser executable (`poser_don_app`) with a single argument, for example:

```
/full/path/to/poser_don_app /full/path/to/some_folder/poser_request.yaml  <poser_response_filename.yaml>
```

And then `poser_don_app` would create write the results to `<poser_response_filename.yaml>`. The third argument is optional, if it is not specified then the results are written to a file named `response.yaml` in the current directory.

## Data input format

The description of the input format in a `poser_request.yaml` is:

```
object_1_name:
  template: /path/to/model_1.pcd
  image_1:
    descriptor_img: /path/to/img.npy
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    save_processed_cloud: /path/to/save[.pcd/.ply]
    save_template: /path/to/save[.pcd/.ply]
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
    descriptor_img: /path/to/img.npy
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    save_processed_cloud: /path/to/save[.pcd/.ply]
    save_template: /path/to/save[.pcd/.ply]
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





One note is that the `mask_img`, `visualize`, `save_processed_cloud` and `save_template` fields are optional, as detailed below. By example here is a proposed valid `poser_request.yaml` file:

```
shoe_1:
  template: /home/wei/shoe_model.pcd
  image_1:
    descriptor_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/descriptor_images/000001_depth.png
    rgb_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_rgb.png
    depth_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/processed/images/000001_depth.png
    mask_img: /home/wei/pdc/logs_proto/2018-04-06-11-34-13/mask_rcnn/000001/mask_001.png
    save_processed_cloud: /home/wei/Coding/poser/data/processed_cloud_world.pcd
    save_template: /home/wei/Coding/poser/data/template.ply
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
- If the user specifies a path for `save_processed_cloud`, the `poser_don_app` will save the cropped, subsampled depth point cloud expressed in **world frame** to the given path for further processing.
- If the user specifies a path for `save_template`, the `poser_don_app` will save the template (currently geometry only) to the given path for further processing.

#### Notes on the Mask and Bounding Box
Internally `poser` crops the pointcloud to a bounding box before doing any estimation, see [these](https://github.com/RobotLocomotion/poser/blob/master/apps/poser_don/preprocessing.cpp#L154) lines. If you omit the `mask_img` field then it defaults to not applying any mask, and only using the bounding box.

## Data output format

The output `poser_out.yaml` copies all of the input data (for redundancy and no ".yaml correspondence" problem) and also adds the computed result:

```
object_1_name:
  template: /path/to/model_1.pcd
  image_1:
    descriptor_img: /path/to/img.npy
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
     affine_transform: # column major 4 x 4 matrix
     rigid_transform: # column major 4 x 4 matrix
object_2_name:
  template: /path/to/model_2.pcd
  image_1:
    descriptor_img: /path/to/img.npy
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
    affine_transform: # column major 4 x 4 matrix
    rigid_transform: # column major 4 x 4 matrix
```

The transforms outputted transform the model to the observation, so they are `T_observation_model`.


# Keypoint Detection


## Input Format
```
object_1_name:
  template: /path/to/model_1.pcd (optional)
  image_1:
    descriptor_img: /path/to/img.npy (optional)
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    save_processed_cloud: /path/to/save[.pcd/.ply] (optional)
    save_template: /path/to/save[.pcd/.ply] (optional)
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

## Output Format
```
object_1_name:
  template: /path/to/model_1.pcd (optional)
  image_1:
    descriptor_img: /path/to/img.npy (optional)
    rgb_img: /path/to/img.png
    depth_img: /path/to/img.png
    mask_img: /path/to/img.png
    save_processed_cloud: /path/to/save[.pcd/.ply] (optional)
    save_template: /path/to/save[.pcd/.ply] (optional)
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
     keypoints_world_frame:
      <keypoint_name_1>: # name
        position: [0.03947, -0.00042, -0.04980]

      <keypoint_name_2>:
        position: [0.00083, 0.04006, -0.04980]

      <keypoint_name_3>:
        position: [-0.03764, -0.00052, -0.04998]

      <keypoint_name_4>:
        position: [0.00233, -0.03790, -0.05000]
```

## How the run mankey Inference

Example data are provided in `${mankey_root}/experiment/inference_pdc_data` , please modify the path in `mankey_request.yaml` according to your directory structure. 

Prepare the network weight, which will be used as a script argument. An example network weight can be downloaded at [here](https://drive.google.com/file/d/1fVWa4I2DcnApESc1RYvRjMSCg2lq2uSG/view?usp=sharing).

Once everything is ready, run

```shell
cd ${mankey_root}
export PYTHONPATH="${PWD}:${PYTHONPATH}"
cd ${mankey_root}/experiment
python inference_pdc.py --request_path path/to/request.yaml --response_path path/to/response.yaml --network_chkpt_path path/to/network/checkpoint.pth
```

Note that the python interpreter should be python3 in virtualenv. If everything is OK, you should see 3d visualization window (if enabled) and the keypoint in world frame will be written to response.yaml