from future.utils import iteritems

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
import dense_correspondence_manipulation.utils.utils as pdc_utils
import dense_correspondence.loss_functions.utils as loss_utils

import dense_correspondence.loss_functions.loss_functions as loss_functions

# compute pixel match error
COMPUTE_PIXEL_MATCH_ERROR = True

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
                            verbose=False,
                            seed=1,
                            ):

    pdc_utils.reset_random_seed(seed)

    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    pdc_utils.saveToYaml(config, os.path.join(train_dir, "config.yaml"))

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
    print("use_gpu", use_gpu)

    """
    Build model
    """
    descriptor_dim = config['network']['descriptor_dimension']
    fcn_model = fcn_resnet("fcn_resnet101", descriptor_dim, pretrained=True)
    # fcn_model = fcn_resnet("fcn_resnet50", descriptor_dim, pretrained=True)
    model = DenseDescriptorNetwork(fcn_model, normalize=False)

    # setup optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=config['train']['lr'], betas=(config['train']['adam_beta1'], 0.999))
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  patience=config['train']['lr_scheduler_patience'],
                                  factor=config['train']['lr_scheduler_factor'],
                                  verbose=True)

    # loss criterion
    if use_gpu:
        model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)


    # mse_loss
    mse_loss = torch.nn.MSELoss()
    heatmap_type = config['loss_function']['heatmap_type']

    # const
    best_valid_pixel_error = np.inf
    global_iteration = 0
    counters = {'train': 0, 'valid': 0}
    st_epoch = 0
    epoch_counter_external = 0

    # setup some values to be assigned later
    sigma = None
    image_diagonal_pixels = None

    # setup meter losses
    meter_loss = {'heatmap_loss': AverageMeter(),
                  'best_match_pixel_error': AverageMeter(),
                  }


    try:
        for epoch in range(st_epoch, config['train']['n_epoch']):
            phases = ['train', 'valid']
            epoch_counter_external = epoch

            writer.add_scalar("Training Params/epoch", epoch, global_iteration)
            for phase in phases:
                model.train(phase == 'train')

                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]

                # reset the meter_loss
                for _, meter in iteritems(meter_loss):
                    meter.reset()

                for i, data in enumerate(loader):

                    global_iteration += 1
                    counters[phase] += 1

                    with torch.set_grad_enabled(phase == 'train'):

                        if verbose:
                            print("\n\n------global iteration %d-------" %(global_iteration))

                        # compute the descriptor images
                        # [B, 3, H, W]
                        rgb_tensor_a = data['data_a']['rgb_tensor'].to(device)
                        rgb_tensor_b = data['data_b']['rgb_tensor'].to(device)

                        B, _, H, W = rgb_tensor_a.shape

                        # compute sigma for heatmap loss
                        image_diagonal_pixels = np.sqrt(H**2 + W**2)
                        sigma = image_diagonal_pixels * config['loss_function']['sigma_fraction']

                        out_a = model.forward(rgb_tensor_a)
                        out_b = model.forward(rgb_tensor_b)

                        if verbose:
                            print("camera_name_a", data['data_a']['camera_name'])
                            print("camera_name_b", data['data_b']['camera_name'])

                        des_img_a = out_a['descriptor_image']
                        des_img_b = out_b['descriptor_image']

                        # compute the loss
                        # [B,2,N]
                        uv_a = data['matches']['uv_a'].to(device)
                        uv_b = data['matches']['uv_b'].to(device)
                        valid = data['matches']['valid'].to(device)

                        _, _, N = uv_a.shape

                        # [B,N,D]
                        des_a = pdc_utils.index_into_batch_image_tensor(des_img_a, uv_a).permute([0,2,1])


                        if verbose:
                            print("des_a.shape", des_a.shape)
                            print("des_img_b.shape", des_img_b.shape)


                        # this is the heatmap that an individual descriptor from rgb_a
                        # will induce given descriptor image b (des_img_b)
                        # [B,N,H,W]
                        heatmap_pred = loss_utils.compute_heatmap_from_descriptors(des_a,
                                                                                   des_img_b,
                                                                                   sigma=sigma,
                                                                                   type=heatmap_type)
                        # [M, H, W]
                        heatmap_pred_valid = pdc_utils.extract_valid(heatmap_pred, valid=valid)

                        # compute ground truth heatmap
                        # [M, 2]
                        uv_b_valid = pdc_utils.extract_valid(uv_b.permute([0,2,1]), valid)

                        # [M, H, W]
                        heatmap_gt = loss_utils.create_heatmap(uv_b_valid,
                                                               H=H,
                                                               W=W,
                                                               sigma=sigma,
                                                               type=heatmap_type)

                        # if verbose:
                        #     print("heatmap_gt.shape", heatmap_gt.shape)
                        #     print("heatmap_pred_valid.shape", heatmap_pred_valid.shape)
                        #     print("heatmap_gt.device", heatmap_gt.device)
                        #     print("uv_b_valid.device", uv_b_valid.device)

                        # compute an L2 heatmap loss
                        heatmap_l2_loss = mse_loss(heatmap_pred_valid, heatmap_gt)
                        loss = heatmap_l2_loss

                        if COMPUTE_PIXEL_MATCH_ERROR:

                            # [B, N, D, H, W]
                            expand_batch_des_a = pdc_utils.expand_descriptor_batch(des_a, H, W)
                            expand_des_img_b = pdc_utils.expand_image_batch(des_img_b, N)

                            # [B, N, H, W]
                            norm_diff = (expand_batch_des_a - expand_des_img_b).norm(p=2, dim=2)


                            best_match_dict = pdc_utils.find_pixelwise_extreme(norm_diff, type="min")


                            # [B, N, 2]
                            uv_b_pred = best_match_dict['indices']
                            uv_b_gt = uv_b.permute([0, 2, 1])
                            pixel_diff = (uv_b_pred - uv_b_gt).type(torch.float)

                            # [B, N]
                            pixel_error = torch.norm(pixel_diff, dim=2)

                            # [M, 2]
                            pixel_error_valid = pdc_utils.extract_valid(pixel_error, valid)

                            # as a percentage of image diagonal
                            avg_pixel_error = torch.mean(pixel_error_valid)
                            avg_pixel_error_percent = avg_pixel_error.item() * 100/image_diagonal_pixels

                            median_pixel_error = torch.median(pixel_error_valid)
                            median_pixel_error_percent = median_pixel_error*100/image_diagonal_pixels


                    # update meter losses
                    meter_loss['heatmap_loss'].update(loss.item())
                    meter_loss['best_match_pixel_error'].update(avg_pixel_error_percent.item())


                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if i % config['train']['log_per_iter'] == 0:
                        log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                            phase, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))

                        log += ', pixel_error: %.6f' % (avg_pixel_error.item())
                        log += ', meter_avg_pixel_error: %.6f' % (meter_loss['best_match_pixel_error'].avg)

                        print(log)

                        # log data to tensorboard
                        # only do it once we have reached 500 iterations
                        if global_iteration > 200:
                            if phase == "train":
                                writer.add_scalar("Params/learning rate", get_lr(optimizer), counters[phase])

                            writer.add_scalar("Loss/%s" %(phase), loss.item(), counters[phase])

                            if COMPUTE_PIXEL_MATCH_ERROR:
                                writer.add_scalar("pixel_match_error/mean/%s" % (phase), avg_pixel_error.item(),
                                                  counters[phase])
                                writer.add_scalar("pixel_match_error/median/%s" % (phase), median_pixel_error.item(),
                                                  counters[phase])

                                writer.add_scalar("pixel_match_error_percent/mean/%s" % (phase), avg_pixel_error_percent.item(),
                                                  counters[phase])
                                writer.add_scalar("pixel_match_error_percent/median/%s" % (phase), median_pixel_error_percent.item(),
                                                  counters[phase])


                    if phase == 'train' and i % config['train']['ckp_per_iter'] == 0:
                        save_model(model, '%s/net_dy_epoch_%d_iter_%d' % (train_dir, epoch, i))


                writer.add_scalar("avg_pixel_error/%s" %(phase), meter_loss['best_match_pixel_error'].avg, epoch)

                if phase == 'valid':

                    # scheduler.step(meter_loss['heatmap_loss'].avg)
                    if meter_loss['best_match_pixel_error'].avg < best_valid_pixel_error:
                        best_valid_pixel_error = meter_loss['best_match_pixel_error'].avg
                        save_model(model, '%s/net_best_dy' % (train_dir))

                writer.flush() # flush SummaryWriter events to disk

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model, '%s/net_dy_epoch_%d_keyboard_interrupt' % (train_dir, epoch_counter_external))
        writer.flush() # flush SummaryWriter events to disk






