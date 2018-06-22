# Pytorch Dense Correspondence inside Docker

## Quickstart

The following is all of the steps to build `pdc` with docker from a fresh Ubuntu installation:

1) Install [Docker for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
  - Make sure to `sudo usermod -aG docker your-user` and then not run below docker scripts as `sudo`
2) Install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker). Make sure to use `nvidia-docker1` not `nvidia-docker2` as it has a known issue with OpenGL. See [this](https://github.com/RobotLocomotion/spartan/issues/201) issue. Follow the instructions on their webpage but replace
```
sudo apt-get install -y nvidia-docker2
```
with
```
sudo apt-get install -y nvidia-docker
```
You can test that your nvidia-docker installation is working by running
```
nvidia-docker run --rm nvidia/cuda nvidia-smi
```
If you get errors about nvidia-modprobe not being installed, install it by running
```
sudo apt-get install nvidia-modprobe
```
and then restart your machine.

3) Clone, setup, and build `pdc`: You need to have ssh keys setup to clone the submodules. Make sure that these ssh keys don't have a password, otherwise it will not work.
```
git clone git@github.com:RobotLocomotion/pytorch-dense-correspondence.git
cd pytorch-dense-correspondence
git submodule sync
git submodule update --init --remote
git submodule update
cd docker
./docker_build.py
```

Now there should be a docker image called `<username>-pytorch-dense-correspondence` on your machine
Below is explained additional options and details of the above.

## Docker Cheatsheet

Handling images
- `docker images` - lists all docker images on machine, including REPOSITORY, TAG, IMAGE_ID, when created, size
- `docker tag IMAGE_ID NEW_NAME` - creates a new REPOSITORY:TAG for an IMAGE_ID
- `docker rmi REPOSITORY:TAG` - removes this tag for an image
- `docker tag IMAGE_ID my-spartan && docker rmi spartan` -- example to combine above two commands to rename an image ID

Handling containers
- `docker ps -a` - lists all containers on machine
- `docker rm CONTAINER_ID` - removes container id 
