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

Some of the 3D vision processing lives in Spartan, some lives in pytorch-dense-correspondence.

#### In Spartan
When done collecting raw data, just run `./batch_extract_and_fuse_all_scenes.py`.

This will run all TSDF fusion and mesh generation.

#### In pytorch-dense-correspondence
Start docker container (`cd docker && ./docker_run.py`), navigate to `pdc/logs_proto`, then:
```
use_pytorch_dense_correspondence
use_director
run_change_detection_pipeline.py
```
This will run all change detection, masking, and synthetic depth rendering.

## Training

1. Start by grabbing data (use `pdc-scripts` repo for internal users to transfer data)
2. Make a `config/dense_correspondence/dataset/your_dataset.yaml`
3. Start (or use already open) docker container (`cd docker && ./docker_run.py`)
4. Open jupyter notebook, which is used for various visualizations (`./start_notebook.py`), then open a browser window with the token it spews out.
5. Recommend checking this dataset out by running `datset/simple_datasets_test.ipynb` and looking at debug visuals
6. Compute the mean of this dataset
7. Make choices in `config/dense_correspondence/training/training.yaml`
8. Run training via `dense_correspondence/training/training.ipynb`.  Note may want to choose your `cuda_visible_devices` in this notebook, or set a global flag for this in `config/defaults.yaml`.
9. Open and view visdom via navigating a browser window to `0.0.0.0:8097` (this will be live after training begins)

All outputs from training will be saved in `pdc/trained_models` in a folder particular to this training.  Unless you specify differently in `training.yaml`, this will save to `pdc/trained_models/test/` folder.

## Evaluation

1. Run a qualitative eval (see `evaluation_plots_example.ipynb`)
2. Run a quantitative eval (see `evaluation_quantitative.ipynb`)
3. Plot the results of quantitative eval (see `evaluation_quantitative_plots.ipynb`)
4. If looking ready for a cross-scene evaluation, then label cross-scene data with `simple-pixel-correspondence-labeler`
5. Run cross-scene qualitative eval (see `evaluation_plots_cross_scene.ipynb`)
6. Run cross-scene quantitative eval (see `evaluation_quantitative_cross_scene.ipynb`)
7. Plot the results of quantitative eval, can do both same-scene and cross-scene together (see `evaluation_quantitative_plots.ipynb`)
