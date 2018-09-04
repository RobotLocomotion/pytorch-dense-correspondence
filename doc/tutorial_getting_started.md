## Tutorial: Getting started with pytorch-dense-correspondence

#### From zero to finishing a dense descriptor network training in 30 minutes or less


This guide is meant to walk you through how to start from scratch and start training dense descriptor networks for objects.
We'll use the data and approach from our [paper, "Dense Object Nets"](https://arxiv.org/abs/1806.08756).


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

We have a dedicated separate page for how to build a docker container for this project.  See [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/docker_build_instructions.md),
and when you're done, head on back to this page.  While the docker image is building (may take a handful of minutes), you can start on the next step.
