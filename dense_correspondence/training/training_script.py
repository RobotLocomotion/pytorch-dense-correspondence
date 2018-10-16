import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

#utils.set_default_cuda_visible_devices()
# utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'shoe_train_1_red_nike.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "code/data_volume/pdc/trained_models/2018-10-15/"
num_iterations = 800
d = 2 # the descriptor dimension
name = "shoes_progress_iterative_%d" %(d)
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations

TRAIN = True
EVALUATE = True

# All of the saved data for this network will be located in the
# code/data_volume/pdc/trained_models/tutorials/caterpillar_3 folder


### NON ITERATIVE

if TRAIN:
    print "training descriptor of dimension %d" %(d)
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
    print "finished training descriptor of dimension %d" %(d)


quit()

### ITERATIVE

num_iterations = num_iterations/4

# First 
config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'shoe_train_1_green_nike.yaml')
config = utils.getDictFromYamlFilename(config_filename)
dataset = SpartanDataset(config=config)

print "training descriptor of dimension %d" %(d)
train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
train.run()
print "finished training descriptor of dimension %d" %(d)

# Second 
config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'shoe_train_1_gray_nike.yaml')
config = utils.getDictFromYamlFilename(config_filename)
dataset = SpartanDataset(config=config)

print "training descriptor of dimension %d" %(d)
train_config["training"]["logging_dir_name"] = name+"1"
train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
train.run_from_pretrained("2018-10-15/"+name)
print "finished training descriptor of dimension %d" %(d)

# Third 
config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'shoe_train_1_red_nike.yaml')
config = utils.getDictFromYamlFilename(config_filename)
dataset = SpartanDataset(config=config)

print "training descriptor of dimension %d" %(d)
train_config["training"]["logging_dir_name"] = name+"2"
train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
train.run_from_pretrained("2018-10-15/"+name+"1")
print "finished training descriptor of dimension %d" %(d)

# Fourth
config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'shoe_train_1_brown_boot.yaml')
config = utils.getDictFromYamlFilename(config_filename)
dataset = SpartanDataset(config=config)

print "training descriptor of dimension %d" %(d)
train_config["training"]["logging_dir_name"] = name+"3"
train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
train.run_from_pretrained("2018-10-15/"+name+"2")
print "finished training descriptor of dimension %d" %(d)