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
    print("use_gpu", use_gpu)

    """
    Build model
    """
    descriptor_dim = 3
    non_match_margin = 0.5
    fcn_model = fcn_resnet("fcn_resnet101", descriptor_dim, pretrained=True)
    # fcn_model = fcn_resnet("fcn_resnet50", descriptor_dim, pretrained=True)
    model = DenseDescriptorNetwork(fcn_model, normalize=False)

    # setup optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=config['train']['lr'], betas=(config['train']['adam_beta1'], 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    # loss criterion

    if use_gpu:
        model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)


    # mse_loss
    mse_loss = torch.nn.MSELoss()

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

                        if verbose:
                            print("\n\n------global iteration %d-------" %(global_iteration))

                        # compute the descriptor images
                        rgb_tensor_a = data['data_a']['rgb_tensor'].to(device)
                        rgb_tensor_b = data['data_b']['rgb_tensor'].to(device)
                        out_a = model.forward(rgb_tensor_a)
                        out_b = model.forward(rgb_tensor_b)

                        if verbose:
                            print("camera_name_a", data['data_a']['camera_name'])
                            print("camera_name_b", data['data_b']['camera_name'])

                        des_img_a = out_a['descriptor_image']
                        des_img_b = out_b['descriptor_image']

                        # compute the loss
                        uv_a = data['matches']['uv_a'].to(device)
                        uv_b = data['matches']['uv_b'].to(device)
                        valid = data['matches']['valid'].to(device)

                        des_a = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_a, uv_a, valid)['des']
                        # print("des_a.shape", des_a.shape)
                        # print("valid.shape", valid.shape)
                        # num_valid = torch.nonzero(valid).shape[0]
                        # print("num_valid", num_valid)
                        # raise ValueError("TEST")

                        des_b = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_b, uv_b, valid)['des']


                        # compute an L2 match loss
                        match_loss = mse_loss(des_a, des_b)


                        # # masked non-match loss
                        mnm_uv_a = data['masked_non_matches']['uv_a'].to(device)
                        mnm_uv_b = data['masked_non_matches']['uv_b'].to(device)
                        mnm_valid = data['masked_non_matches']['valid'].to(device)

                        mnm_des_a = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_a, mnm_uv_a, mnm_valid)['des']

                        mnm_des_b = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_b, mnm_uv_b, mnm_valid)['des']

                        mnm_loss_dict = loss_functions.non_match_loss(mnm_des_a, mnm_des_b, non_match_margin)
                        masked_non_match_loss = mnm_loss_dict['loss']

                        # background non-match-loss
                        bnm_uv_a = data['background_non_matches']['uv_a'].to(device)
                        bnm_uv_b = data['background_non_matches']['uv_b'].to(device)
                        bnm_valid = data['background_non_matches']['valid'].to(device)

                        bnm_des_a = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_a, bnm_uv_a, bnm_valid)['des']
                        bnm_des_b = pdc_utils.index_into_batch_image_tensor_and_extract_valid(des_img_b, bnm_uv_b, bnm_valid)['des']

                        bnm_loss_dict = loss_functions.non_match_loss(bnm_des_a, bnm_des_b, non_match_margin)
                        background_non_match_loss = bnm_loss_dict['loss']

                        loss = match_loss + 0.5*masked_non_match_loss + 0.5*background_non_match_loss


                        if COMPUTE_PIXEL_MATCH_ERROR:
                            # compute best match
                            # read dimensions
                            B = rgb_tensor_a.shape[0]
                            D = rgb_tensor_a.shape[1]
                            H = rgb_tensor_a.shape[2]
                            W = rgb_tensor_a.shape[3]
                            N = uv_a.shape[2]

                            # [B, D, N]
                            batch_des_a = pdc_utils.index_into_batch_image_tensor(des_img_a, uv_a)

                            # [B, N, D]
                            batch_des_a = batch_des_a.permute([0, 2, 1])

                            # [B, N , D, H, W]
                            expand_batch_des_a = pdc_utils.expand_descriptor_batch(batch_des_a, H, W)
                            expand_des_img_b = pdc_utils.expand_image_batch(des_img_b, N)

                            # [B, N, H, W]
                            norm_diff = (expand_batch_des_a - expand_des_img_b).norm(p=2, dim=2)

                            # if verbose:
                            #     print("norm_diff.shape", norm_diff.shape)


                            best_match_dict = pdc_utils.find_pixelwise_extreme(norm_diff,
                                                                               type="min")

                            # [B, 2, N]
                            best_match_uv_b = best_match_dict['indices'].permute([0,2,1])

                            # if verbose:
                            #     print("best_match_uv_b.shape", best_match_uv_b.shape)

                            # compute distance to GT
                            # [B, N]
                            # requires allocating [B, N, D, H, W]
                            pixel_error = (uv_a - best_match_uv_b).type(torch.float).norm(p=2, dim=1)

                            # if verbose:
                            #     print("pixel_error.shape", pixel_error.shape)

                            # get extract valid ones
                            # [M]
                            valid_pixel_error = pixel_error.flatten()[valid.flatten() > 0]

                            # if verbose:
                            #     print("valid_pixel_error.shape", valid_pixel_error.shape)

                            pixel_error_mean = torch.mean(valid_pixel_error)
                            pixel_error_median = torch.median(valid_pixel_error)

                            if verbose:
                                print("pixel_error_mean:", pixel_error_mean.item())
                                print("pixel_error_median:", pixel_error_median.item())


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
                        if global_iteration > 200:
                            writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                            writer.add_scalar("Loss/train", loss.item(), global_iteration)


                            writer.add_scalar("Match_Loss/%s" %(phase), match_loss.item(), global_iteration)
                            writer.add_scalar("Masked_non_match_loss/%s" %(phase), masked_non_match_loss.item(), global_iteration)
                            writer.add_scalar("Background_non_match_loss/%s" %(phase), background_non_match_loss.item(), global_iteration)

                            if COMPUTE_PIXEL_MATCH_ERROR:
                                writer.add_scalar("pixel_match_error/mean/%s" %(phase), pixel_error_mean.item(), global_iteration)
                                writer.add_scalar("pixel_match_error/median/%s" %(phase), pixel_error_median.item(), global_iteration)

                            # writer.add_scalar("RMSE average loss/train", meter_loss_rmse.avg, global_iteration)

                    if phase == 'train' and i % config['train']['ckp_per_iter'] == 0:
                        save_model(model, '%s/net_dy_epoch_%d_iter_%d' % (train_dir, epoch, i))


                #
                # log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                #     phase, epoch, config['train']['n_epoch'], meter_loss_rmse.avg, best_valid_loss)
                # print(log)

                if phase == 'valid':
                    pass
                    # writer.add_scalar("Loss/valid", loss.item(), global_iteration)
                    # writer.add_scalar("Match_Loss/valid", match_loss.item(), global_iteration)
                    # writer.add_scalar("Masked_non_match_loss/valid", masked_non_match_loss.item(), global_iteration)
                    # writer.add_scalar("Background_non_match_loss/valid", background_non_match_loss.item(),
                    #                   global_iteration)


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






