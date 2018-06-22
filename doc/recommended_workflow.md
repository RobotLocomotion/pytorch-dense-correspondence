# Recommended Workflow

This doc outlines the recommended workflow for `pdc`. You always work from inside a docker container.

## Setting the path to your data directory
Edit `config/docker_run_config.yaml` to set the path to your data volume. This will be mounted inside the docker at the location `~/code/data_volume`.

## Launching a docker container
You should already have built the docker image as outlined [here](docker_build_instructions.md). To launch the container

```
cd docker
./docker_run.py
```
## Inside the docker container
You are now inside the docker container. If do `terminator &` you will launch a new terminator with a blue background.

- source the necessary environment variables with `use_pytorch_dense_correspondence`. If using scripts that rely on `director` then also run `use_director`.
- start a Jupyter notebook with the `~/code/start_notebook.py`.
