# Recommended Workflow

This doc outlines the recommended workflow for `pytorch-dense-correspondence`.

The typical workflow we use is:

- edit code which is outside the docker container (so you can use whichever text editor / IDE you like) but that is externally mounted into the docker container
- to run code, first launch a Jupyter notebook server from inside the docker container
- run code and visualize data in web browser via Jupyter notebook

## One time setup: setting the path to your data directory
Edit `config/docker_run_config.yaml` to set the path to your data volume. By data volume we mean the folder that contains the `pdc` data folder as outline in the [data organization](data_organization.md) doc. This will be mounted inside the docker at the location `~/code/data_volume`.

## Launching a docker container 
You should already have built the docker image as outlined [here](docker_build_instructions.md). To launch the container

```
cd docker
./docker_run.py
```
## Launching a Jupyter notebook server inside the docker container
You are now inside the docker container. Run `terminator &` you will launch a new terminator with a blue background.

- source the necessary environment variables with `use_pytorch_dense_correspondence`. If using scripts that rely on `director` then also run `use_director`.
- start a Jupyter notebook server with the `~/code/start_notebook.py`.

## Interacting via Jupyter notebook

The terminal from which you ran the `start_notebook.py` command will print out a URL.

Navigate there to interact via Jupyter.

---

### Docker Cheatsheet

You won't need to know much about docker to work with this repo, since the image can remain untouched once it's built, but here are a few quick commands to know for deleting unused images and containers:

Handling images
- `docker images` - lists all docker images on machine, including REPOSITORY, TAG, IMAGE_ID, when created, size
- `docker rmi IMAGE_ID` - removes this image

Handling containers
- `docker ps -a` - lists all containers on machine
- `docker rm CONTAINER_ID` - removes container id 
