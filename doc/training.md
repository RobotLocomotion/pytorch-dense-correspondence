
# Training a Dense Correspondence Network

## DenseCorrespondenceTraining

The class `DenseCorrespondenceTraining` (in `training.py`) manages the training process. It takes in a config specifying parameters. See `config/dense_correspondence/training/training.yaml` for an example. You just need to call `run()` on the `DenseCorrespondenceTraining` object to run the training. Plotting is done using Visdom, navigate to `localhost:8097` in the browser and choose the appropriate `env` from the dropdown menu for your training run. Note that the env will be the same as the name of the logging directory for that training run.


## Example
See the notebook `training.ipynb` for an example of how to use this class.
