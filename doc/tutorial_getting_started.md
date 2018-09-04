## Tutorial: Getting started with pytorch-dense-correspondence

#### From zero to finishing a dense descriptor network training in 30 minutes or less


This guide is meant to walk you through how to start from scratch and start training dense descriptor networks for objects.
We'll use the data and approach from our [paper, "Dense Object Nets"](https://arxiv.org/abs/1806.08756).

## Step 0: Clone the repo

```
git clone https://github.com/RobotLocomotion/pytorch-dense-correspondence.git
```

## Step 1: Download the data

Decide where you want to download the data to and then use our provided script to 
download a subset of the data.  (This script will only download a 7.8 GB subset of data.  The full dataset is about 100 GB of data.)

```
## first, navigate to where you want to download the data
## this example will just place the data folder inside of pytorch-dense-correspondence
cd pytorch-dense-correspondence
python config/download_pdc_data.py config/dense_correspondence/dataset/composite/caterpillar_only.yaml
```

The above will download only the data for the scenes of data with the caterpillar object, as a starting subest of the data.

Note that the data is all downloaded into a folder called `pdc`.  These three letters signify the start of the standardized dataset formatting.

While the data is downloading, you can move on through the next few steps.

## Step 2: Configure where your data is

Edit the file `config/docker_run_config.yaml` to point to the directory that contains where the data (i.e., the directory above `pdc`):

For example if your username is `username` and your hostname is `hostname`, you would add an entry for:

```
hostname:
  username:
    path_to_data_directory: '/home/username/pytorch-dense-correspondence/'
```

## Step 3: Build the docker image

We have a dedicated separate page for how to build a docker image for this project.  See [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/docker_build_instructions.md),
and when you're done, head on back to this page.  While the docker image is building (will take a handful of minutes), you can start on the next step.

## Step 4: Make sure that the permissions of .torch are for your user

This one-line script will ensure this is OK:

```
sudo chown -R $USER:$USER ~/.torch
```

## Step 5: Run the docker image and start a jupyter notebook server

```
cd pytorch-dense-correspondence/docker
./docker_run.py
terminator ## this will pop open a new window, with a blue background, so you know you're in the docker container
use_pytorch_dense_correspondence ## this sets useful environment variables
./start_notebook.py
```

The jupyter notebook will direct you to point a browser window (Chrome/Firefox/etc) to something like:

`127.0.0.1:8888/?token=603eeac08495233c8f08abbf4caa2c5124da2864ae6f8103`

## Step 6: Start training a network and evaluate it quantitatively

Open the notebook for training, `dense_correspondence/training/training_tutorial.ipynb`.

Run each of the cells in the notebook.

The final cell will run the quantitative evaluation.

## Step 7: Qualitatively evaluate the network

Open the notebook for qualitative evaluation, `dense_correspondence/evaluation/evaluation_qualitative_tutorial.ipynb`.

Run each of the cells here!

