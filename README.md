## Dense Correspondence Learning in PyTorch

The aim of this repo is to provide tools for dense correspondence learning in PyTorch.  In particular:

- Implementation of components used in for "Self-supervised Visual Descriptor Learning for Dense Correspondence" by T. Schmidt, R. A. Newcombe, D. Fox
- Training scripts to train models
- Integration with open-source RGBD fusion (ElasticFusion)
  
  
### Code Setup

- [building with docker](doc/docker_build_instructions.md)
- [recommended docker workflow ](doc/recommended_workflow.md)


### Dataset

- [data organization](doc/data_organization.md)
- [data processing for a single scene](doc/data_processing_single_scene.md)
- [data processing in batch](doc/data_processing_batch.md)


### Dense Object Nets
- [recommended docker workflow ](doc/recommended_workflow.md)
- [training a network](doc/training.md)
- [evaluating a trained network](doc/dcn_evaluation.md)


### Miscellaneous
- [coordinate conventions](doc/coordinate_conventions.md)

### Tutorials

### Git management

To prevent the repo from growing in size, recommend always "restart and clear outputs" before committing any Jupyter notebooks.  If you'd like to save what your notebook looks like, you can always "download as .html", which is a great way to snapshot the state of that notebook and share.
