# Pytorch Dense Correspondence inside Docker

## Introduction

We highly recommend using a docker environment to work with this repo.  All development for this project has used this setup.

The docker image essentially packages all dependencies in a safe environment.  The scripts we provide will externally mount our source code, and our data, into the docker environment.

Most source code for this project is in Python and so once the docker image is built we won't need any compiling.

## Quickstart

The following is all of the steps to build a docker image for `pytorch-dense-correspondence` from a fresh Ubuntu installation:

1) Install [Docker for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
  - Make sure to `sudo usermod -aG docker your-user` and then not run below docker scripts as `sudo`
2) Install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker). You can test that your nvidia-docker installation is working by running
```
nvidia-docker run --rm nvidia/cuda nvidia-smi
```
If you get errors about nvidia-modprobe not being installed, install it by running
```
sudo apt-get install nvidia-modprobe
```
and then restart your machine.

**Note**: It's possible that the latest nvidia-docker doesn't include `nvidia-smi`. Try testing your `docker` installation by running
```
nvidia-docker run --rm nvidia/cuda:9.1-base nvidia-smi.
```
instead.

3) Clone, setup, and build docker image for `pytorch-dense-correspondence`. If using clone via `ssh`, you need to have ssh keys setup to clone the submodules. Make sure that these ssh keys don't have a password, otherwise it will not work.  Cloning via `https` should be OK.
```
git clone git@github.com:RobotLocomotion/pytorch-dense-correspondence.git
cd pytorch-dense-correspondence
git submodule update --init --recursive
cd docker
./docker_build.py
```

You're done with setup!

Now there should be a docker image called `<username>-pytorch-dense-correspondence` on your machine.
