## Tutorial: Getting started with pytorch-dense-correspondence

#### From zero to finishing a dense descriptor network training in 30 minutes or less

This guide is meant to walk you through how to start from scratch and start training dense descriptor networks for objects.
We'll use the data and approach from our [paper, "Dense Object Nets"](https://arxiv.org/abs/1806.08756).

## Requirements

- An Ubuntu machine (we've tested 14.04, 16.04, 18.04) with an Nvidia GPU (recommended at least 4 GB of memory)
- Python 2 and a handful of Python 2 modules (`yaml`, etc) to run initial scripts
- Everything else will be set up automatically inside an `nvidia-docker` container

## Step 0: Clone the repo

```
git clone https://github.com/RobotLocomotion/pytorch-dense-correspondence.git
```

## Step 1: Download the data

Decide where you want to download the data to and then use our provided script to 
download a subset.  (This script will only download a 5.3 GB subset of data.  The full dataset is about 100 GB.)

```
## first, navigate to where you want to download the data
## this example will just place the data folder inside of pytorch-dense-correspondence
## you may alternatively want to place the data on an external drive
cd pytorch-dense-correspondence
python config/download_pdc_data.py config/dense_correspondence/dataset/composite/caterpillar_only_9.yaml
```

The above will download only the scenes with the single caterpillar object, as a starting subset of the data.

*Note that the data is all downloaded into a folder called `pdc`.*  These three letters signify the start of a structured dataset subdirectory hierarchy. See our [documentation on data organization to learn more.](data_organization.md)

While the data is downloading, you can move on through the next few steps.

## Step 2: Configure where your data is

Edit the file `config/docker_run_config.yaml` to point to the directory that contains the directory above `pdc`.

For example if your username is `username` and your hostname is `hostname`, and you just put the `pdc` folder inside of `pytorch-dense-correspondence`, you would add an entry for:

```
hostname:
  username:
    path_to_data_directory: '/path/to/pytorch-dense-correspondence/'
```

## Step 3: Build the docker image

If you already have `nvidia-docker` installed, then you can just run:

```
cd pytorch-dense-correspondence
git submodule update --init --recursive
cd docker
./docker_build.py
```

If instead you are new to `nvidia-docker`, we have a dedicated separate page for how to build a docker image for this project.  See [here](docker_build_instructions.md),
and when you're done, head on back to this page.  

While the docker image is building (will take a handful of minutes), you can start on the next step.

## Step 4: Make sure that the permissions of .torch are for your user

This one-line script will ensure this is OK:

```
mkdir -p ~/.torch && sudo chown -R $USER:$USER ~/.torch
```

## Step 5: Run the docker image and start a jupyter notebook server

```
cd pytorch-dense-correspondence/docker
./docker_run.py
terminator ## this will pop open a new window, with a blue background, so you know you're in the docker container
use_pytorch_dense_correspondence ## this sets necessary environment variables
./start_notebook.py
```

The output from the jupyter notebook server will direct you to point a browser window (Chrome/Firefox/etc) to something like:

`http://127.0.0.1:8888/?token=603eeac08495233c8f08abbf4caa2c5124da2864ae6f8103`

## Step 6: Run a simple dataset loader test

Head to the notebook [`dense_correspondence/dataset/simple_datasets_test.ipynb`](../dense_correspondence/dataset/simple_datasets_test.ipynb) and run through the cells.  This is a simple test to make sure all the data is configured properly.  If you see a bunch of correspondence visualizations, you're good!  Head on to the next step to start training.

## Step 7: Start training a network and evaluate it quantitatively

Note: Make sure the data download from Step 1 has finished before continuing.

Open the notebook for training, [`dense_correspondence/training/training_tutorial.ipynb`](../dense_correspondence/training/training_tutorial.ipynb).

Run each of the cells in the notebook.  The final cell will run the quantitative evaluation.  To visualize the plots from the quantitative evaluation, go to [`dense_correspondence/evaluation/evaluation_quantitative_tutorial.ipynb`](../dense_correspondence/evaluation/evaluation_quantitative_tutorial.ipynb) which is set up to compare the results of N >= 1 networks.

## Step 8: Qualitatively evaluate the network

Open the notebook for qualitative evaluation, [`dense_correspondence/evaluation/evaluation_qualitative_tutorial.ipynb`](../dense_correspondence/evaluation/evaluation_qualitative_tutorial.ipynb). 

Run each of the cells here to see descriptor plots!

## Step 9: View a heatmap visualization

In a new terminal (you can split your docker terminator window with Ctrl+Shift+E):

```
use_pytorch_dense_correspondence
cd modules/user-interaction-heatmap-visualization
python live_heatmap_visualization.py
```

This brings up the heatmap visualization. Using the learned descriptors for the caterpillar network you just trained it finds correspondences between the source and target images. The best match for the point you are mousing over in the target image is indicated by the reticle in the target image. Other nearby points (in descriptor space) are indicated by the dark regions in the grayscale image.

Use `n` on your keyboard to bring up a new pair of random images.

---

## Getting Help
If you run into any issues during the tutorial please create an issue and we will try our best to help you resolve it! We also welcome suggestions for improvements if you found any of the steps confusing.

## What next?

Now that you've gone through the into tutorial, here are a few suggestions on things you could try next.

- Try more data.  You can download our entire processed dataset with this command: `python config/download_pdc_data.py config/dense_correspondence/dataset/composite/entire_dataset.yaml`

- Try your own architecture / training procedures / loss function.

- Run more more analysis and visualizations:
  - Heatmaps of correspondences: in `modules/user-interaction-heatmap-visualization/live_heatmap_visualization.py`
  - Plot scatter plots of samples in descriptor space: in `dense_correspondence/evaluation/evaluation_clusters_2d.ipynb`
  - Make a video in descriptor space: in `dense_correspondence/evaluation/make_video.ipynb`
