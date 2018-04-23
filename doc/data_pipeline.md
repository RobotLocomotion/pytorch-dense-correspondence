## Data pipeline

The purpose of this doc is to provide an overview of the entire process from:

- **Raw data collection**  (capturing raw RGBD video and forward kinematics poses)
- **3D vision processing** (TSDF fusion and mesh generation, change detection and masking, synthetic depth rendering off meshes)
- **Training** (Training a dense descriptor network off this data)
- **Evaluation** (Evaluating the performance of the dense descriptor network)

## Raw data collection

For us, all raw data collection happens in [`spartan`](https://github.com/RobotLocomotion/spartan).  

- *Note*: While Spartan is all open source, it's a little customized to our particular robot + RGBD sensing setup.  It can be useful as reference for outside users, but it's not our intent to make it user-friendly.  But once you have your RGBD data with pose estimates of choice, `pytorch-dense-correspondence` is intended to be well documented and useful for a variety of projects!

The quick version of raw data collection currently is:

#### Human-moved objects

1. Start Kuka, run Position control
2. `kip` (shortcut for Kuka Iiwa Procman) then in procman: 
    1. Start ROS script (check that openni driver looks happy)
    2. Run Kuka drivers
    3. Check that pointcloud / sensor data in RVIZ looks OK
3. New terminal: prepare to collect logs via `use_ros && use_spartan`, navigate to fusion server scripts (`cd spartan/src/catkin_projects/fusion_server/scripts`)
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

## 3D vision processing
   
