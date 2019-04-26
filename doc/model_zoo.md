# Model Zoo
This document contains links to some pre-trained models. See [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/doc/dcn_evaluation.md)
for how to load and visualize a pre-trained network. Prior to evaluating make sure you have downloaded the appropriate train
and test data as outlined in the tutorial.

Inside of the folders for each of these pre-trained networks you will find: 
- a `dataset.yaml`, which documents which dataset configuration was used
- a `training.yaml`, which documents which training configuration was used
- an `analysis.yaml`, which contains quantitative analysis and is plottable using the `evaluation_quantitative_tutorial.ipynb` notebook
- the `tensorboard`subdirectory containing all info needed to plot the results of the training
- a handful of other files, including network weights at various iterations


Models

  1. [Caterpillar](https://data.csail.mit.edu/labelfusion/pdccompressed/trained_models/stable/caterpillar_standard_params_3.tar.gz) -- note that this network was trained with M_background = 2.0 and M_masked = 0.5, this means there is a larger contrastive loss with the background than there is with the rest of the object.  While we didn't find this to improve quantitative visualizations, it can improve qualitative visualizations of the descriptor space.
  2. [Shoes Class Consistent](https://data.csail.mit.edu/labelfusion/pdccompressed/trained_models/stable/shoes_consistent_M_background_0.500_3.tar.gz)
  3. [Hats Class Consistent](https://data.csail.mit.edu/labelfusion/pdccompressed/trained_models/stable/hats_consistent_M_background_0.500_3.tar.gz)
