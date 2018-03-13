## Dense Correspondence Learning in PyTorch

The aim of this repo is to provide tools for dense correspondence learning in PyTorch.  In particular:

- Implementation of components used in for "Self-supervised Visual Descriptor Learning for Dense Correspondence" by T. Schmidt, R. A. Newcombe, D. Fox
- Training scripts to train models
- Integration with open-source RGBD fusion (ElasticFusion)
  
  
### Setup

First get the repo and all submodules:

```
git clone https://github.com/peteflorence/pytorch-dense-correspondence.git
cd pytorch-dense-correspondence
git submodule update --init --recursive
```

Recommend using provided Docker scripts to run (install `nvidia-docker` if you haven't):

```
cd docker
./docker_build.py
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
