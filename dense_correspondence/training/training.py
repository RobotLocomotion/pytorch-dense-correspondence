# system
import numpy as np
import os
import fnmatch
import gc
import logging
import time
import shutil
import subprocess
import copy

# torch
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import visdom
import tensorboard_logger
from torchnet.logger import VisdomPlotLogger, VisdomLogger



# dense correspondence
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
import pytorch_segmentation_detection.models.fcn as fcns
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from pytorch_segmentation_detection.transforms import (ComposeJoint,
                                                       RandomHorizontalFlipJoint,
                                                       RandomScaleJoint,
                                                       CropOrPad,
                                                       ResizeAspectRatioPreserve,
                                                       RandomCropJoint,
                                                       Split2D)

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import dense_correspondence.loss_functions.loss_composer as loss_composer
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation


class DenseCorrespondenceTraining(object):

    def __init__(self, config=None, dataset=None, dataset_test=None):
        if config is None:
            config = DenseCorrespondenceTraining.load_default_config()

        self._config = config
        self._dataset = dataset
        self._dataset_test = dataset_test

        self._dcn = None
        self._optimizer = None

    def setup(self):
        """
        Initializes the object
        :return:
        :rtype:
        """
        self.load_dataset()
        self.setup_logging_dir()
        self.setup_visdom()
        self.setup_tensorboard()


    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def load_dataset(self):
        """
        Loads a dataset, construct a trainloader.
        Additionally creates a dataset and DataLoader for the test data
        :return:
        :rtype:
        """

        batch_size = self._config['training']['batch_size']
        num_workers = self._config['training']['num_workers']

        if self._dataset is None:
            self._dataset = SpartanDataset.make_default_10_scenes_drill()

        
        self._dataset.load_all_pose_data()
        self._dataset.set_parameters_from_training_config(self._config)

        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

        # create a test dataset
        if self._config["training"]["compute_test_loss"]:
            if self._dataset_test is None:
                self._dataset_test = SpartanDataset(mode="test", config=self._dataset.config)

            
            self._dataset_test.load_all_pose_data()
            self._dataset_test.set_parameters_from_training_config(self._config)

            self._data_loader_test = torch.utils.data.DataLoader(self._dataset_test, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

    def load_dataset_from_config(self, config):
        """
        Loads train and test datasets from the given config
        :param config: Dict gotten from a YAML file
        :type config:
        :return: None
        :rtype:
        """
        self._dataset = SpartanDataset(mode="train", config=config)
        self._dataset_test = SpartanDataset(mode="test", config=config)
        self.load_dataset()

    def build_network(self):
        """
        Builds the DenseCorrespondenceNetwork
        :return:
        :rtype: DenseCorrespondenceNetwork
        """

        return DenseCorrespondenceNetwork.from_config(self._config['dense_correspondence_network'],
                                                      load_stored_params=False)

    def _construct_optimizer(self, parameters):
        """
        Constructs the optimizer
        :param parameters: Parameters to adjust in the optimizer
        :type parameters:
        :return: Adam Optimizer with params from the config
        :rtype: torch.optim
        """

        learning_rate = float(self._config['training']['learning_rate'])
        weight_decay = float(self._config['training']['weight_decay'])
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _get_current_loss(self, logging_dict):
        """
        Gets the current loss for both test and train
        :return:
        :rtype: dict
        """
        d = dict()
        d['train'] = dict()
        d['test'] = dict()

        for key, val in d.iteritems():
            for field in logging_dict[key].keys():
                vec = logging_dict[key][field]

                if len(vec) > 0:
                    val[field] = vec[-1]
                else:
                    val[field] = -1 # placeholder


        return d

    def load_pretrained(self, model_folder, iteration=None):
        """
        Loads network and optimizer parameters from a previous training run.

        Note: It is up to the user to ensure that the model parameters match.
        e.g. width, height, descriptor dimension etc.

        :param model_folder: location of the folder containing the param files 001000.pth. Can be absolute or relative path. If relative then it is relative to pdc/trained_models/
        :type model_folder:
        :param iteration: which index to use, e.g. 3500, if None it loads the latest one
        :type iteration:
        :return: iteration
        :rtype:
        """

        if not os.path.isdir(model_folder):
            pdc_path = utils.getPdcPath()
            model_folder = os.path.join(pdc_path, "trained_models", model_folder)

        # find idx.pth and idx.pth.opt files
        if iteration is None:
            files = os.listdir(model_folder)
            model_param_file = sorted(fnmatch.filter(files, '*.pth'))[-1]
            iteration = int(model_param_file.split(".")[0])
            optim_param_file = sorted(fnmatch.filter(files, '*.pth.opt'))[-1]
        else:
            prefix = utils.getPaddedString(iteration, width=6)
            model_param_file = prefix + ".pth"
            optim_param_file = prefix + ".pth.opt"

        print "model_param_file", model_param_file
        model_param_file = os.path.join(model_folder, model_param_file)
        optim_param_file = os.path.join(model_folder, optim_param_file)


        self._dcn = self.build_network()
        self._dcn.load_state_dict(torch.load(model_param_file))
        self._dcn.cuda()
        self._dcn.train()

        self._optimizer = self._construct_optimizer(self._dcn.parameters())
        self._optimizer.load_state_dict(torch.load(optim_param_file))

        return iteration

    def run_from_pretrained(self, model_folder, iteration=None, learning_rate=None):
        """
        Wrapper for load_pretrained(), then run()
        """
        iteration = self.load_pretrained(model_folder, iteration)
        if iteration is None:
            iteration = 0

        if learning_rate is not None:
            self._config["training"]["learning_rate_starting_from_pretrained"] = learning_rate
            self.set_learning_rate(self._optimizer, learning_rate)

        self.run(loss_current_iteration=iteration, use_pretrained=True)

    def run(self, loss_current_iteration=0, use_pretrained=False):
        """
        Runs the training
        :return:
        :rtype:
        """

        start_iteration = copy.copy(loss_current_iteration)

        DCE = DenseCorrespondenceEvaluation

        self.setup()
        self.save_configs()

        if not use_pretrained:
            # create new network and optimizer
            self._dcn = self.build_network()
            self._optimizer = self._construct_optimizer(self._dcn.parameters())
        else:
            logging.info("using pretrained model")
            if (self._dcn is None):
                raise ValueError("you must set self._dcn if use_pretrained=True")
            if (self._optimizer is None):
                raise ValueError("you must set self._optimizer if use_pretrained=True")

        # make sure network is using cuda and is in train mode
        dcn = self._dcn
        dcn.cuda()
        dcn.train()

        optimizer = self._optimizer
        batch_size = self._data_loader.batch_size

        pixelwise_contrastive_loss = PixelwiseContrastiveLoss(image_shape=dcn.image_shape, config=self._config['loss_function'])
        pixelwise_contrastive_loss.debug = True

        loss = match_loss = non_match_loss = 0

        max_num_iterations = self._config['training']['num_iterations'] + start_iteration
        logging_rate = self._config['training']['logging_rate']
        save_rate = self._config['training']['save_rate']
        compute_test_loss_rate = self._config['training']['compute_test_loss_rate']

        # logging
        self._logging_dict = dict()
        self._logging_dict['train'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "masked_non_match_loss": [], 
                                           "background_non_match_loss": [],
                                           "blind_non_match_loss": [],
                                           "learning_rate": [],
                                           "different_object_non_match_loss": []}

        self._logging_dict['test'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "non_match_loss": []}

        # save network before starting
        if not use_pretrained:
            self.save_network(dcn, optimizer, 0)

        for epoch in range(50):  # loop over the dataset multiple times

            for i, data in enumerate(self._data_loader, 0):
                loss_current_iteration += 1
                start_iter = time.time()

                match_type, \
                img_a, img_b, \
                matches_a, matches_b, \
                masked_non_matches_a, masked_non_matches_b, \
                background_non_matches_a, background_non_matches_b, \
                blind_non_matches_a, blind_non_matches_b, \
                metadata = data

                if (match_type == -1).all():
                    print "\n empty data, continuing \n"
                    continue


                data_type = metadata["type"][0]
                
                img_a = Variable(img_a.cuda(), requires_grad=False)
                img_b = Variable(img_b.cuda(), requires_grad=False)

                matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_a = Variable(masked_non_matches_a.cuda().squeeze(0), requires_grad=False)
                masked_non_matches_b = Variable(masked_non_matches_b.cuda().squeeze(0), requires_grad=False)

                background_non_matches_a = Variable(background_non_matches_a.cuda().squeeze(0), requires_grad=False)
                background_non_matches_b = Variable(background_non_matches_b.cuda().squeeze(0), requires_grad=False)

                blind_non_matches_a = Variable(blind_non_matches_a.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_b = Variable(blind_non_matches_b.cuda().squeeze(0), requires_grad=False)

                optimizer.zero_grad()
                self.adjust_learning_rate(optimizer, loss_current_iteration)

                # run both images through the network
                image_a_pred = dcn.forward(img_a)
                image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

                image_b_pred = dcn.forward(img_b)
                image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

                # get loss
                loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss = loss_composer.get_loss(pixelwise_contrastive_loss, match_type,
                                                                                image_a_pred, image_b_pred,
                                                                                matches_a,     matches_b,
                                                                                masked_non_matches_a, masked_non_matches_b,
                                                                                background_non_matches_a, background_non_matches_b,
                                                                                blind_non_matches_a, blind_non_matches_b)
                

                loss.backward()
                optimizer.step()

                elapsed = time.time() - start_iter

                print "single iteration took %.3f seconds" %(elapsed)


                def update_visdom_plots(loss, match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss):
                    """
                    Updates the visdom plots with current loss function information
                    :return:
                    :rtype:
                    """

                    learning_rate = DenseCorrespondenceTraining.get_learning_rate(optimizer)
                    self._logging_dict['train']['learning_rate'].append(learning_rate)
                    self._visdom_plots['learning_rate'].log(loss_current_iteration, learning_rate)
                    self._tensorboard_logger.log_value("learning rate", learning_rate, loss_current_iteration)


                    # Don't update any plots if the entry corresponding to that term
                    # is a zero loss
                    if not loss_composer.is_zero_loss(match_loss):
                        self._logging_dict['train']['match_loss'].append(match_loss.data[0])
                        self._visdom_plots['train']['match_loss'].log(loss_current_iteration, match_loss.data[0])
                        self._tensorboard_logger.log_value("train match loss", match_loss.data[0], loss_current_iteration)

                    if not loss_composer.is_zero_loss(masked_non_match_loss):
                        self._logging_dict['train']['masked_non_match_loss'].append(masked_non_match_loss.data[0])
                        self._visdom_plots['train']['masked_non_match_loss'].log(loss_current_iteration,
                                                             masked_non_match_loss.data[0])
                        self._tensorboard_logger.log_value("train masked non match loss", masked_non_match_loss.data[0], loss_current_iteration)

                    if not loss_composer.is_zero_loss(background_non_match_loss):
                        self._logging_dict['train']['background_non_match_loss'].append(background_non_match_loss.data[0])
                        self._visdom_plots['train']['background_non_match_loss'].log(loss_current_iteration,
                                                             background_non_match_loss.data[0])
                        self._tensorboard_logger.log_value("train background non match loss", background_non_match_loss.data[0], loss_current_iteration)

                    if not loss_composer.is_zero_loss(blind_non_match_loss):

                        if data_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
                            self._tensorboard_logger.log_value("train blind SINGLE_OBJECT_WITHIN_SCENE", blind_non_match_loss.data[0], loss_current_iteration)

                        if data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
                            self._tensorboard_logger.log_value("train blind DIFFERENT_OBJECT", blind_non_match_loss.data[0], loss_current_iteration)


                    # loss is never zero
                    if data_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
                        self._tensorboard_logger.log_value("train loss SINGLE_OBJECT_WITHIN_SCENE", loss.data[0], loss_current_iteration)

                    elif data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
                        self._tensorboard_logger.log_value("train loss DIFFERENT_OBJECT", loss.data[0], loss_current_iteration)

                    elif data_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
                        self._tensorboard_logger.log_value("train loss SINGLE_OBJECT_ACROSS_SCENE", loss.data[0], loss_current_iteration)

                    elif data_type == SpartanDatasetDataType.MULTI_OBJECT:
                        self._tensorboard_logger.log_value("train loss MULTI_OBJECT", loss.data[0], loss_current_iteration)
                    else:
                        raise ValueError("unknown data type")


                    if data_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
                        self._tensorboard_logger.log_value("train different object", loss.data[0], loss_current_iteration)

                    # #non_match_type = metadata['non_match_type'][0]
                    # fraction_hard_negatives = pixelwise_contrastive_loss.debug_data['fraction_hard_negatives']

                    # if pixelwise_contrastive_loss.debug:
                    #     if non_match_type == "masked":
                    #         self._visdom_plots['masked_hard_negative_rate'].log(loss_current_iteration, fraction_hard_negatives)
                    #         self._tensorboard_logger.log_value("masked hard negative rate", fraction_hard_negatives, loss_current_iteration)
                    #     elif non_match_type == "non_masked":
                    #         self._visdom_plots['non_masked_hard_negative_rate'].log(loss_current_iteration,
                    #                                                             fraction_hard_negatives)

                    #         self._tensorboard_logger.log_value("non-masked hard negative rate", fraction_hard_negatives,
                    #                                      loss_current_iteration)
                    #     else:
                    #         raise ValueError("uknown non_match_type %s" %(non_match_type))


                # def update_visdom_test_loss_plots(test_loss, test_match_loss, test_non_match_loss):
                #     """
                #     Log data about test loss and update the visdom plots
                #     :return:
                #     :rtype:
                #     """

                #     self._logging_dict['test']['loss'].append(test_loss)
                #     self._logging_dict['test']['match_loss'].append(test_match_loss)
                #     self._logging_dict['test']['non_match_loss'].append(test_non_match_loss)
                #     self._logging_dict['test']['iteration'].append(loss_current_iteration)


                #     self._visdom_plots['test']['loss'].log(loss_current_iteration, test_loss)
                #     self._visdom_plots['test']['match_loss'].log(loss_current_iteration, test_match_loss)
                #     self._visdom_plots['test']['non_match_loss'].log(loss_current_iteration, test_non_match_loss)

                #     self._tensorboard_logger.log_value('test loss', test_loss, loss_current_iteration)
                #     self._tensorboard_logger.log_value('test match loss', test_match_loss, loss_current_iteration)
                #     self._tensorboard_logger.log_value('test non-match loss', test_non_match_loss, loss_current_iteration)



                update_visdom_plots(loss, match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss)

                if loss_current_iteration % save_rate == 0:
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)

                if loss_current_iteration % logging_rate == 0:
                    logging.info("Training on iteration %d of %d" %(loss_current_iteration, max_num_iterations))

                    logging.info("single iteration took %.3f seconds" %(elapsed))

                    percent_complete = loss_current_iteration * 100.0/(max_num_iterations - start_iteration)
                    logging.info("Training is %d percent complete\n" %(percent_complete))


                # don't compute the test loss on the first few times through the loop
                if self._config["training"]["compute_test_loss"] and (loss_current_iteration % compute_test_loss_rate == 0) and loss_current_iteration > 5:
                    logging.info("Computing test loss")

                    # delete the loss, match_loss, non_match_loss variables so that
                    # pytorch can use that GPU memory
                    del loss, match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss
                    gc.collect()

                    dcn.eval()
                    test_loss, test_match_loss, test_non_match_loss = DCE.compute_loss_on_dataset(dcn,
                                                                                                  self._data_loader_test, self._config['loss_function'], num_iterations=self._config['training']['test_loss_num_iterations'])

                    update_visdom_test_loss_plots(test_loss, test_match_loss, test_non_match_loss)

                    # delete these variables so we can free GPU memory
                    del test_loss, test_match_loss, test_non_match_loss

                    # make sure to set the network back to train mode
                    dcn.train()

                if loss_current_iteration % self._config['training']['garbage_collect_rate'] == 0:
                    logging.debug("running garbage collection")
                    gc_start = time.time()
                    gc.collect()
                    gc_elapsed = time.time() - gc_start
                    logging.debug("garbage collection took %.2d seconds" %(gc_elapsed))

                if loss_current_iteration > max_num_iterations:
                    logging.info("Finished testing after %d iterations" % (max_num_iterations))
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)
                    return


    def setup_logging_dir(self):
        """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
        """

        if 'logging_dir_name' in self._config['training']:
            dir_name = self._config['training']['logging_dir_name']
        else:
            dir_name = utils.get_current_time_unique_name() +"_" + str(self._config['dense_correspondence_network']['descriptor_dimension']) + "d"

        self._logging_dir_name = dir_name

        self._logging_dir = os.path.join(utils.convert_to_absolute_path(self._config['training']['logging_dir']), dir_name)



        if os.path.isdir(self._logging_dir):
            shutil.rmtree(self._logging_dir)

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        # make the tensorboard log directory
        self._tensorboard_log_dir = os.path.join(self._logging_dir, "tensorboard")
        if not os.path.isdir(self._tensorboard_log_dir):
            os.makedirs(self._tensorboard_log_dir)

        return self._logging_dir

    def save_network(self, dcn, optimizer, iteration, logging_dict=None):
        """
        Saves network parameters to logging directory
        :return:
        :rtype: None
        """

        network_param_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + ".pth")
        optimizer_param_file = network_param_file + ".opt"
        torch.save(dcn.state_dict(), network_param_file)
        torch.save(optimizer.state_dict(), optimizer_param_file)

        # also save loss history stuff
        if logging_dict is not None:
            log_history_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + "_log_history.yaml")
            utils.saveToYaml(logging_dict, log_history_file)

            current_loss_file = os.path.join(self._logging_dir, 'loss.yaml')
            current_loss_data = self._get_current_loss(logging_dict)

            utils.saveToYaml(current_loss_data, current_loss_file)



    def save_configs(self):
        """
        Saves config files to the logging directory
        :return:
        :rtype: None
        """
        training_params_file = os.path.join(self._logging_dir, 'training.yaml')
        utils.saveToYaml(self._config, training_params_file)

        dataset_params_file = os.path.join(self._logging_dir, 'dataset.yaml')
        utils.saveToYaml(self._dataset.config, dataset_params_file)        

    def adjust_learning_rate(self, optimizer, iteration):
        """
        Adjusts the learning rate according to the schedule
        :param optimizer:
        :type optimizer:
        :param iteration:
        :type iteration:
        :return:
        :rtype:
        """

        steps_between_learning_rate_decay = self._config['training']['steps_between_learning_rate_decay']
        if iteration % steps_between_learning_rate_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self._config["training"]["learning_rate_decay"]

    @staticmethod
    def set_learning_rate(optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        return lr

    def setup_visdom(self):
        """
        Sets up visdom visualizer
        :return:
        :rtype:
        """
        self.start_visdom()
        self._visdom_env = self._logging_dir_name
        self._vis = visdom.Visdom(env=self._visdom_env)

        self._port = 8097
        self._visdom_plots = dict()

        self._visdom_plots["train"] = dict()
        self._visdom_plots['train']['loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Train Loss'}, env=self._visdom_env)

        self._visdom_plots['learning_rate'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Learning Rate'}, env=self._visdom_env)

        self._visdom_plots['train']['match_loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Train Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['masked_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Masked Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['background_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Background Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['blind_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Blind Non Match Loss'}, env=self._visdom_env)


        self._visdom_plots["test"] = dict()
        self._visdom_plots['test']['loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Loss'}, env=self._visdom_env)

        self._visdom_plots['test']['match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Match Loss'}, env=self._visdom_env)

        self._visdom_plots['test']['non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['masked_hard_negative_rate'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Masked Matches Hard Negative Rate'}, env=self._visdom_env)

        self._visdom_plots['non_masked_hard_negative_rate'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Non-Masked Hard Negative Rate'}, env=self._visdom_env)

    def setup_tensorboard(self):
        """
        Starts the tensorboard server and sets up the plotting
        :return:
        :rtype:
        """

        # start tensorboard
        # cmd = "python -m tensorboard.main"
        logging.info("setting up tensorboard_logger")
        cmd = "tensorboard --logdir=%s" %(self._tensorboard_log_dir)
        self._tensorboard_logger = tensorboard_logger.Logger(self._tensorboard_log_dir)
        logging.info("tensorboard logger started")


    @staticmethod
    def load_default_config():
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence',
                                   'training', 'training.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        return config

    @staticmethod
    def make_default():
        dataset = SpartanDataset.make_default_caterpillar()
        return DenseCorrespondenceTraining(dataset=dataset)


    @staticmethod
    def start_visdom():
        """
        Starts visdom if it's not already running
        :return:
        :rtype:
        """

        vis = visdom.Visdom()

        if vis.check_connection():
            logging.info("Visdom already running, returning")
            return


        logging.info("Starting visdom")
        cmd = "python -m visdom.server"
        subprocess.Popen([cmd], shell=True)