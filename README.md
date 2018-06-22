## Dense Correspondence Learning in PyTorch

The aim of this repo is to provide tools for dense correspondence learning in PyTorch.  In particular:

- Implementation of components used in for "Self-supervised Visual Descriptor Learning for Dense Correspondence" by T. Schmidt, R. A. Newcombe, D. Fox
- Training scripts to train models
- Integration with open-source RGBD fusion (ElasticFusion)
  
  
### Code Setup

- [building with docker](doc/docker_build_instructions.md)
- [recommended workflow ](doc/recommended_workflow.md)

First get the repo and all submodules:

```
git clone https://github.com/peteflorence/pytorch-dense-correspondence.git
cd pytorch-dense-correspondence
git submodule update --init --recursive
```

### Data Setup

For the LabelFusion Dataset loader, do this to use an example 5 scenes (this is only ~3 GB):

```
cd pytorch-dense-correspondence
wget https://data.csail.mit.edu/labelfusion/LabelFusionExampleData/mini-labelfusion-drill.tar.gz
tar -xvzf mini-labelfusion-drill.tar.gz
````

You can also use the full LabelFusion dataset (~500 GB) as long as you go and extract depth images from any scenes you'd like to use.

By default the docker script also mounts `/media` so paths can be specified to datasets in `/media`.  You can set up other ways to find your data.

### Recommended Workflow

Recommend using provided Docker scripts to run (install `nvidia-docker` if you haven't):

```
cd docker
./docker_build.py
```

Edit `config/docker_run_config.yaml` to set the path to your data volume. This will be mounted inside the docker at the location `~/code/data_volume`. Launch the docker with

```
./docker_run.py
```

Visualization and interaction is meant to be through Jupyter notebooks, via:

```
./start_notebook.py
## navigate your browser to the URL provided by the Jupyter output in the terminal
```

Then run some notebooks for training, testing, debugging.

### Git management

To prevent the repo from growing in size, recommend always "restart and clear outputs" before committing any Jupyter notebooks.  If you'd like to save what your notebook looks like, you can always "download as .html", which is a great way to snapshot the state of that notebook and share.
