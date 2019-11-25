import os
import random
import numpy as np

# from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence.network.dense_descriptor_network import fcn_resnet, DenseDescriptorNetwork
from dense_correspondence_manipulation.utils.utils import AverageMeter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")

def train_dense_descriptors(config,
                            train_dir,
                            multi_episode_dict=None,
                            ):
    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    # save_yaml(config, os.path.join(train_dir, "config.yaml")

    # make train/test dataloaders
    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = DynamicDrakeSimDataset(config,
                                                 multi_episode_dict)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train']['num_workers'])

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()

    """
    Build model
    """
    descriptor_dim = 3
    fcn_resnet101 = fcn_resnet("fcn_resnet101", descriptor_dim, pretrained=True)
    model = DenseDescriptorNetwork(fcn_resnet101, normalize=False)

    # setup optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=config['train']['lr'], betas=(config['train']['adam_beta1'], 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    # loss criterion

    if use_gpu:
        model.cuda()


    # const
    best_valid_loss = np.inf
    global_iteration = 0
    st_epoch = 0
    epoch_counter_external = 0

    try:
        for epoch in range(st_epoch, config['train']['n_epoch']):
            phases = ['train', 'valid']
            epoch_counter_external = epoch

            writer.add_scalar("Training Params/epoch", epoch, global_iteration)
            for phase in phases:
                model.train(phase == 'train')

                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]
                meter_loss = AverageMeter()

                for i, data in enumerate(loader):

                    global_iteration += 1

                    with torch.set_grad_enabled(phase == 'train'):

                        # compute the descriptor images
                        out_a = model.forward(data['data_a']['rgb_tensor'])
                        out_b = model.forward(data['data_b']['rgb_tensor'])


                        # compute the loss
                        


                    # compute loss, add it to meter_loss
                    meter_loss.update(loss.item())


                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if i % config['train']['log_per_iter'] == 0:
                        log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                            phase, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))
                        # log += ', rmse: %.6f (%.6f)' % (
                        #     np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                        print(log)

                        # log data to tensorboard
                        # only do it once we have reached 500 iterations
                        if global_iteration > 500:
                            writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                            writer.add_scalar("Loss/train", loss.item(), global_iteration)
                            # writer.add_scalar("RMSE average loss/train", meter_loss_rmse.avg, global_iteration)

                    if phase == 'train' and i % config['train']['ckp_per_iter'] == 0:
                        save_model(model, '%s/net_dy_epoch_%d_iter_%d' % (train_dir, epoch, i))


                #
                # log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                #     phase, epoch, config['train']['n_epoch'], meter_loss_rmse.avg, best_valid_loss)
                # print(log)

                if phase == 'valid':
                    pass
                    # scheduler.step(meter_loss_rmse.avg)
                    # writer.add_scalar("RMSE average loss/valid", meter_loss_rmse.avg, global_iteration)
                    # if meter_loss_rmse.avg < best_valid_loss:
                    #     best_valid_loss = meter_loss_rmse.avg
                    #     save_model(model_dy, '%s/net_best_dy' % (train_dir))

                writer.flush() # flush SummaryWriter events to disk

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model, '%s/net_dy_epoch_%d_keyboard_interrupt' % (train_dir, epoch_counter_external))
        writer.flush() # flush SummaryWriter events to disk






