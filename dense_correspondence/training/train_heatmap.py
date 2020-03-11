from future.utils import iteritems

import os
import random
import numpy as np
import time

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
from dense_correspondence_manipulation.utils import torch_utils
import dense_correspondence.loss_functions.utils as loss_utils
from dense_correspondence.network import predict

# compute pixel match error
COMPUTE_PIXEL_MATCH_ERROR = True

COMPUTE_SPATIAL_LOSS = True
COMPUTE_HEATMAP_LOSS = True
MIN_NUM_MATCHES = 10 # require at least 10 matches to run a batch

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
                            ):

    pdc_utils.reset_random_seed(config['train']['random_seed'])

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
                                                 multi_episode_dict,
                                                 phase=phase)

        print("len(multi_episode_dict)", len(multi_episode_dict))
        print("len(datasets[%s])" %(phase), len(datasets[phase]))

        # optionally use the deprecated image_to_tensor transform from DenseObjectNets
        # paper
        if ("normalization" in config['dataset']) and (config['dataset']['normalization'] == "DON"):
            DON_image_to_tensor = torch_utils.get_deprecated_image_to_tensor_transform()
            datasets[phase].rgb_to_tensor_transform(DON_image_to_tensor)


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
    pretrained = config['network']['pretrained']
    backbone = config['network']['backbone']
    # fcn_model = fcn_resnet("fcn_resnet101", descriptor_dim, pretrained=pretrained)
    # fcn_model = fcn_resnet("fcn_resnet50", descriptor_dim, pretrained=True)
    fcn_model = fcn_resnet(backbone, descriptor_dim, pretrained=pretrained)
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
    l1_loss = torch.nn.L1Loss()
    heatmap_type = config['loss_function']['heatmap']['heatmap_type']

    # const
    best_valid_pixel_error = np.inf
    global_iteration = 0
    counters = {'train': 0, 'valid': 0}
    st_epoch = 0
    epoch_counter_external = 0

    # setup some values to be assigned later
    sigma = None
    sigma_descriptor_heatmap = config['network']['sigma_descriptor_heatmap']
    image_diagonal_pixels = None

    # setup meter losses
    meter_loss = {'heatmap_loss': AverageMeter(),
                  'best_match_pixel_error': AverageMeter(),
                  }


    loss_container = dict() # store the most recent losses

    last_log_time = time.time()


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

                    loss_container = dict()  # store the most recent losses

                    with torch.set_grad_enabled(phase == 'train'):

                        if verbose:
                            print("\n\n------global iteration %d-------" %(global_iteration))

                        # this means we have empty data so we should just skip
                        # this iteration of the loop
                        num_valid_matches = torch.sum(valid)
                        if num_valid_matches < MIN_NUM_MATCHES:
                            print(
                                    "num valid matches (%d) was less than required minimum number (%d), skipping this batch" % (
                            num_valid_matches, MIN_NUM_MATCHES))
                            continue

                        # compute the descriptor images
                        # [B, 3, H, W]
                        rgb_tensor_a = data['data_a']['rgb_tensor'].to(device)
                        rgb_tensor_b = data['data_b']['rgb_tensor'].to(device)

                        B, _, H, W = rgb_tensor_a.shape

                        # compute sigma for heatmap loss
                        image_diagonal_pixels = np.sqrt(H**2 + W**2)
                        sigma = image_diagonal_pixels * config['loss_function']['heatmap']['sigma_fraction']


                        # compute descriptor images
                        # out_a = model.forward(rgb_tensor_a)
                        # out_b = model.forward(rgb_tensor_b)
                        # des_img_a = out_a['descriptor_image']
                        # des_img_b = out_b['descriptor_image']

                        # concatenate rgb_tensor_a and rgb_tensor_b into a single tensor
                        # that we pass through the network

                        # [2*B, 3, H, W]
                        rgb_tensor_stack = torch.cat([rgb_tensor_a, rgb_tensor_b])

                        out = model.forward(rgb_tensor_stack)
                        des_img_stack = out['descriptor_image']

                        # [B, D, H, W]
                        des_img_a = des_img_stack[:B]
                        des_img_b = des_img_stack[B:]

                        # note: It is possible that H_out != H
                        # if we aren't doing the heatmaps at full resolution
                        # in that case it is necessary to downsample the indexing appropriately
                        _, _, H_out, W_out = des_img_stack.shape

                        if verbose:
                            print("camera_name_a", data['data_a']['camera_name'])
                            print("camera_name_b", data['data_b']['camera_name'])


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



                        """
                        Compute the L2 heatmap loss
                        """
                        # this is the heatmap that an individual descriptor from rgb_a
                        # will induce given descriptor image b (des_img_b)
                        # [B,N,H,W]
                        heatmap_pred = loss_utils.compute_heatmap_from_descriptors(des_a,
                                                                                   des_img_b,
                                                                                   sigma=sigma,
                                                                                   type=heatmap_type)
                        # [M, H, W]
                        heatmap_pred_valid = pdc_utils.extract_valid(heatmap_pred, valid=valid)

                        # [M, 2]
                        uv_b_valid = pdc_utils.extract_valid(uv_b.permute([0, 2, 1]), valid)

                        # compute L2 heatmap loss
                        heatmap_l2_loss = None
                        if COMPUTE_HEATMAP_LOSS:
                            # [M, H, W]
                            heatmap_gt = loss_utils.create_heatmap(uv_b_valid,
                                                                   H=H,
                                                                   W=W,
                                                                   sigma=sigma,
                                                                   type=heatmap_type)

                            # compute an L2 heatmap loss
                            heatmap_l2_loss = mse_loss(heatmap_pred_valid, heatmap_gt)
                            loss_container['heatmap'] = heatmap_l2_loss


                        # compute spatial expectation prediction
                        # [M, 2] in uv ordering
                        spatial_expectation_loss = None
                        spatial_expectation_loss_scaled = None
                        if COMPUTE_SPATIAL_LOSS:
                            # this is L1 loss in pixel space
                            # scale it by image diagonal

                            # [M, 2] in uv ordering
                            # note: this step is pretty computationally expensive . . .
                            uv_b_spatial_pred = predict.get_integral_preds_2d(heatmap_pred_valid, verbose=False)


                            # note we multiply by 2 since nn.L1Loss is dividing by the total
                            # number of elements in uv_b_valid (namely 2*M) since it is using
                            # reduction='mean'. To get L1 metric in pixel space we just multiply
                            # the result by 2
                            spatial_expectation_loss = 2*l1_loss(uv_b_valid.type(torch.float), uv_b_spatial_pred)

                            # scale by the image diagonal to keep it invariant to the size of the
                            # output image
                            spatial_expectation_loss_scaled = spatial_expectation_loss/image_diagonal_pixels

                            loss_container['spatial_expectation'] = spatial_expectation_loss_scaled

                            # compute L2 pixel error between uv_b_valid and uv_b_spatial_pred
                            # ultimately this is the number we care about. This is just L2 norm
                            # vs. L1 norm
                            pixel_diff_spatial = (uv_b_valid).type(torch.float) - uv_b_spatial_pred
                            spatial_expectation_pixel_error = torch.mean(torch.norm(pixel_diff_spatial, dim=1))
                            spatial_expectation_pixel_error_percent = spatial_expectation_pixel_error * 100/image_diagonal_pixels



                        # sum up the loss functions
                        loss = 0
                        if config['loss_function']['heatmap']['enabled'] and (heatmap_l2_loss is not None):
                            weight = config['loss_function']['heatmap']['weight']
                            loss += weight * heatmap_l2_loss

                        if config['loss_function']['spatial_expectation']['enabled'] and (spatial_expectation_loss_scaled is not None):
                            weight = config['loss_function']['spatial_expectation']['weight']
                            loss += weight * spatial_expectation_loss_scaled

                        if COMPUTE_PIXEL_MATCH_ERROR:
                            """
                            Computes pixel match error if we use the argmax way of finding
                            the best match
                            """

                            best_match_dict = predict.get_argmax_l2(des_a, des_img_b)

                            # [B, N, 2]
                            uv_b_pred = best_match_dict['indices']
                            uv_b_gt = uv_b.permute([0, 2, 1])
                            pixel_diff = (uv_b_pred - uv_b_gt).type(torch.float)

                            # [B, N]
                            pixel_error = torch.norm(pixel_diff, dim=2)

                            # [M, 2]
                            pixel_error_valid = pdc_utils.extract_valid(pixel_error, valid)

                            # need to make sure this isn't empty
                            if len(pixel_error_valid) > 0:
                                # as a percentage of image diagonal
                                avg_pixel_error = torch.mean(pixel_error_valid)
                                avg_pixel_error_percent = avg_pixel_error.item() * 100/image_diagonal_pixels

                                median_pixel_error = torch.median(pixel_error_valid)
                                median_pixel_error_percent = median_pixel_error*100/image_diagonal_pixels


                    # update meter losses
                    meter_loss['heatmap_loss'].update(loss.item())
                    meter_loss['best_match_pixel_error'].update(avg_pixel_error.item())


                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if i % config['train']['log_per_iter'] == 0:
                        log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                            phase, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))

                        log += ', pixel_error: %.1f' % (avg_pixel_error.item())
                        # log += ', meter_avg_pixel_error: %.6f' % (meter_loss['best_match_pixel_error'].avg)
                        log += ', spatial_pixel_error: %.1f' % (spatial_expectation_pixel_error.item())
                        log += ', spatial_expectation_loss: %.1f' %(spatial_expectation_loss.item())

                        # print how log it took to do 10 steps
                        elapsed = time.time() - last_log_time
                        last_log_time = time.time()
                        log += ", elapsed: %.2f" %(elapsed)

                        print(log)


                        # log data to tensorboard
                        # only do it once we have reached 500 iterations
                        if global_iteration > 200:
                            if phase == "train":
                                writer.add_scalar("Params/learning rate", get_lr(optimizer), counters[phase])

                            writer.add_scalar("Loss/%s" %(phase), loss.item(), counters[phase])

                            for loss_type, loss_obj in loss_container.items():
                                plot_name = "Loss/%s/%s" %(loss_type, phase)
                                writer.add_scalar(plot_name, loss_obj.item(), counters[phase])

                            if COMPUTE_SPATIAL_LOSS:
                                writer.add_scalar("spatial_pixel_match_error/%s" %(phase), spatial_expectation_pixel_error.item(), counters[phase])
                                writer.add_scalar("spatial_pixel_match_error_percent/%s" % (phase),
                                                  spatial_expectation_pixel_error_percent.item(), counters[phase])

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






