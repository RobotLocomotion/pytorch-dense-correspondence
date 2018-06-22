# Data Pipeline for a Single Scene

This outlines how to collect and process data for a single scene. The folder structure (as will be explained below) is 
```
scene_name/
  * raw/
    * fusion.bag
  * processed/
    * fusion_mesh.ply
    * fusion_pointcloud.ply
    * tsdf.bin
    * images/
      * 000000_rgb.png
      * 000000_depth.png
      *
      *
    * rendered_images/
      * 000000_depth.png
      * 000000_depth_cropped.png
      *
      *
    * image_masks/
      * 000000_mask.png
      * 000000_visible_mask.png
      *
      *
      
```

## Capture Raw data with Kuka

The quick version of raw data collection currently is:

#### Human-moved objects

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

#### Autonomous robot-moved objects

1. Start Kuka, run Position control
2. `kip` (shortcut for Kuka Iiwa Procman) then in procman: 
    1. Start ROS script (check that openni driver looks happy)
    2. Run Kuka drivers
    3. Check that pointcloud / sensor data in RVIZ looks OK
    4. Run Director
3. In Director terminal (f8), enter: `graspSupervisor.testInteractionLoop()`

This will
