# system
import numpy as np
import os
import logging
import time
import shutil
import subprocess

# torch
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import visdom
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

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss





class DenseCorrespondenceTraining(object):

    def __init__(self, config=None):
        if config is None:
            config = DenseCorrespondenceTraining.load_default_config()

        self._config = config

    def setup(self):
        """
        Initializes the object
        :return:
        :rtype:
        """
        self.load_dataset()
        self.setup_logging_dir()
        self.setup_visdom()


    def load_dataset(self):
        """
        Loads a dataset, construct a trainloader
        :return:
        :rtype:
        """
        batch_size = self._config['training']['batch_size']
        num_workers = self._config['training']['num_workers']

        self._dataset = SpartanDataset(mode="train")
        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

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

    def run(self):
        """
        Runs the training
        :return:
        :rtype:
        """

        self.setup()
        self.save_configs()
        dcn = self.build_network()
        optimizer = self._construct_optimizer(dcn.parameters())
        batch_size = self._data_loader.batch_size

        pixelwise_contrastive_loss = PixelwiseContrastiveLoss()

        loss = match_loss = non_match_loss = 0
        loss_current_iteration = 0



        max_num_iterations = self._config['training']['num_iterations']
        logging_rate = self._config['training']['logging_rate']
        save_rate = self._config['training']['save_rate']

        # logging
        self._logging_dict = dict()
        self._logging_dict['loss_history'] = []
        self._logging_dict['match_loss_history'] = []
        self._logging_dict['non_match_loss_history'] = []
        self._logging_dict['loss_iteration_number_history'] = []

        # save network before starting
        self.save_network(dcn, optimizer, 0)



        for epoch in range(50):  # loop over the dataset multiple times

            for i, data in enumerate(self._data_loader, 0):
                loss_current_iteration += 1
                start_iter = time.time()

                # get the inputs
                data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = data
                data_type = data_type[0]

                if len(matches_a[0]) == 0:
                    print "didn't have any matches, continuing"
                    continue

                img_a = Variable(img_a.cuda(), requires_grad=False)
                img_b = Variable(img_b.cuda(), requires_grad=False)

                if data_type == "matches":
                    matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                    matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                    non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
                    non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)

                optimizer.zero_grad()
                self.adjust_learning_rate(optimizer, loss_current_iteration)


                # run both images through the network
                image_a_pred = dcn.forward(img_a)
                image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

                image_b_pred = dcn.forward(img_b)
                image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

                # get loss
                if data_type == "matches":
                    loss, match_loss, non_match_loss =\
                        pixelwise_contrastive_loss.get_loss(image_a_pred,
                                                            image_b_pred,
                                                            matches_a,
                                                            matches_b,
                                                            non_matches_a,
                                                            non_matches_b)

                loss.backward()
                optimizer.step()


                def update_visdom_plots():
                    """
                    Updates the visdom plots with current loss function information
                    :return:
                    :rtype:
                    """
                    self._logging_dict['loss_iteration_number_history'].append(loss_current_iteration)
                    self._logging_dict['loss_history'].append(loss.data[0])
                    self._logging_dict['match_loss_history'].append(match_loss)
                    self._logging_dict['non_match_loss_history'].append(non_match_loss)

                    self._visdom_plots['train_loss'].log(loss_current_iteration, loss.data[0])
                    self._visdom_plots['match_loss'].log(loss_current_iteration, match_loss)
                    self._visdom_plots['non_match_loss'].log(loss_current_iteration,
                                                             non_match_loss)



                update_visdom_plots()

                if loss_current_iteration % save_rate == 0:
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)

                if loss_current_iteration % logging_rate == 0:
                    logging.info("Training on iteration %d of %d" %(loss_current_iteration, max_num_iterations))

                    percent_complete = loss_current_iteration * 100.0/max_num_iterations
                    logging.info("Training is %d percent complete\n" %(percent_complete))


                if loss_current_iteration > max_num_iterations:
                    logging.info("Finished training after %d iterations" % (max_num_iterations))
                    return


                # loss_history.append(loss.data[0])
                # match_loss_history.append(match_loss)
                # non_match_loss_history.append(non_match_loss)
                # loss_iteration_number_history.append(loss_current_iteration)

                # this is for testing


    def setup_logging_dir(self):
        """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
        """

        if 'logging_dir_name' in self._config['training']:
            dir_name =  self._config['training']['logging_dir_name']
        else:
            dir_name = utils.get_current_time_unique_name() +"_" + str(self._config['dense_correspondence_network']['descriptor_dimension']) + "d"

        self._logging_dir_name = dir_name

        self._logging_dir = os.path.join(utils.convert_to_absolute_path(self._config['training']['logging_dir']), dir_name)



        if os.path.isdir(self._logging_dir):
            shutil.rmtree(self._logging_dir)

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

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



    def save_configs(self):
        """
        Saves config files to the logging directory
        :return:
        :rtype: None
        """
        training_params_file = os.path.join(self._logging_dir, 'training.yaml')
        utils.saveToYaml(self._config, training_params_file)

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
                param_group['lr'] = param_group['lr'] * 0.9


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

        self._visdom_plots['train_loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Train Loss'}, env=self._visdom_env)

        self._visdom_plots['learning_rate'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Learning Rate'}, env=self._visdom_env)

        self._visdom_plots['match_loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Match Loss'}, env=self._visdom_env)

        self._visdom_plots['non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Non Match Loss'}, env=self._visdom_env)


    @staticmethod
    def load_default_config():
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence',
                                   'training', 'training.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        return config

    @staticmethod
    def make_default():
        return DenseCorrespondenceTraining()


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

