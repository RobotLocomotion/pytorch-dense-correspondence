# Data Pipeline for a Single Scene

This outlines how to collect and process data for a single scene. See [here](dataset_organization.md) for how the dataset is organized. The steps here are split across code in two repos.
- [spartan](https://github.com/RobotLocomotion/spartan) handles the raw data collection and tsdf fusion.
- pdc handles change detection and rendering.

## Spartan

### Capture Raw data with Kuka

The quick version of raw data collection currently is:

##### Human-moved objects

1. Start Kuka, run Position control
2. `kip` (shortcut for Kuka Iiwa Procman) then in procman: 
    1. Start ROS script (check that openni driver looks happy)
    2. Run Kuka drivers
    3. Check that pointcloud / sensor data in RVIZ looks OK
3. New terminal: prepare to collect logs via navigate to fusion server scripts:
```
use_ros && use_spartan
cd ~/spartan/src/catkin_projects/fusion_server/scripts
```
4. Collect many raw logs, for each:
    1. Move objects to desired position
    2. `./capture_scene_client.py`
    3. This will create a new folder with the current date (e.g. `2018-04-07-20-23-56`) and the `raw/fusion.bag` file as in the folder structure above.

##### Autonomous robot-moved objects

1. Start Kuka, run Position control
2. `kip` (shortcut for Kuka Iiwa Procman) then in procman: 
    1. Start ROS script (check that openni driver looks happy)
    2. Run Kuka drivers
    3. Check that pointcloud / sensor data in RVIZ looks OK
    4. Run Director
3. In Director terminal (f8), enter: `graspSupervisor.testInteractionLoop()`

### TSDF Fusion
This is done in `spartan`. Navigate to `spartan/src/catkin_projects/fusion_server/scripts`. With `log_dir` set to the directory of your log, i.e. the full path to `2018-04-07-20-23-56` run 

```
./extract_and_fuse_single_scene.py <full_path_to_log_folder>
```


This will

1. Extract all the rgb and depth images into `processed/images`
2. Produces `processed/images/camera_info.yaml` which contains the camera intrinsic information.
2. Produces `processed/images/pose_data.yaml` which contains the camera pose corresponding to each image.
2. Run tsdf fusion
3. Convert the tsdf fusion to a mesh and save it as `processed/fusion_mesh.ply`
4. Downsample the images in `processed/images` and only keep those with poses that are sufficiently different.

## PDC

### Change Detection and Depth Image Rendering
This is done in `pytorch-dense-correspondence`. In `pdc`

1. `use_pytorch_dense_correspondence`
2. `use_director`
3. `run_change_detection --data_dir <full_path_to_log_folder>/processed`

This will run change detection and render new depth images for the full scene and the cropped scene. The data that is produced by this step is

- `processed/rendered_images/000000_depth.png`
- `processed/rendered_images/000000_depth_cropped.png`
- `processed/image_masks/000000_mask.png`
- `processed/image_masks/000000_mask_visible.png`

