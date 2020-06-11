import os
from torchvision import transforms
import torch
import numpy as np

def make_default_image_to_tensor_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([transforms.ToTensor(), normalize])

def get_deprecated_image_to_tensor_transform():
    mean = [0.5573105812072754, 0.37420374155044556, 0.37020164728164673]
    std_dev = [0.24336038529872894, 0.2987397611141205, 0.31875079870224]

    normalize = transforms.Normalize(mean=mean,
                                     std=std_dev)

    return transforms.Compose([transforms.ToTensor(), normalize])



def random_sample_from_masked_image_torch(img_mask, num_samples, without_replacement=True):
    """
    :param img_mask: Numpy array [H,W] or torch.Tensor with shape [H,W]
    :type img_mask:
    :param num_samples: an integer
    :type num_samples:
    :return: torch.LongTensor shape [num_samples, 2] in u,v ordering
    :rtype:
    """

    nonzero_idx = torch.nonzero(img_mask, as_tuple=True)
    num_nonzero = nonzero_idx[0].numel()

    if without_replacement and (num_nonzero < num_samples):
        raise ValueError("insufficient number of non-zero values to sample without replacement")

    sample_idx = torch.randperm(num_nonzero)[:num_samples]
    u_tensor = nonzero_idx[1][sample_idx]
    v_tensor = nonzero_idx[0][sample_idx]
    uv_tensor = torch.stack((u_tensor, v_tensor), dim=1)

    return uv_tensor



def pinhole_unprojection(uv,  # [B, N, 2] uv pixel coordinates
                         z,  # [B, N] depth values (meters) (all valid
                         K_inv,  # [B, 3, 3] camera matrix inverse
                         ): # [B, N, 3] pts in camera frame
    """
    Projects points from pixels (uv) and depth (z) to 3D space in camera frame

    Test by visualizing in simple_dataset_test_episode_reader.ipynb script
    """
    # print("uv.shape", uv.shape)
    # print("z.shape", z.shape)

    # [B, 2, N]
    uv_tmp = uv.transpose(1,2).type_as(z) * z.unsqueeze(1)

    # print("uv_tmp.shape", uv_tmp.shape)

    # [B, 3, N]
    uv_s = torch.cat((uv_tmp, z.unsqueeze(1)), axis=1)


    # print("uv_s.shape", uv_s.shape)
    # print("K_inv.shape", K_inv.shape)

    # [B, 3, N]
    pts = torch.matmul(K_inv.type_as(uv_s), uv_s)

    return pts.transpose(1,2) # return [B, N, 3]


def pinhole_projection(pts, # [B, N, 3] pts in camera frame
                       K, # [B, 3, 3] camera matrix
                       ): # [B, N, 2] uv coordinates in camera frame
    """
    Projects points from camera frame into pixel space


    Test by visualizing in simple_dataset_test_episode_reader.ipynb script
    """

    # [B, N]
    z = pts.select(dim=-1, index=2)

    # [B, 3, N]
    uv_s = torch.matmul(K.type_as(pts), pts.transpose(1,2))

    # [B, 2, N]
    uv = uv_s[:, :2] / z.unsqueeze(1)

    return uv.transpose(1,2)

def transform_points_3D(T, # [B, 4, 4]
                        pts, # [B, N, 3]
                        ): # [B, N, 3]

    B, N, _ = pts.shape

    # [B, N, 4]
    pts_homog = torch.cat((pts, torch.ones([B,N,1], dtype=pts.dtype)), dim=-1)

    # we are multiplying two tensors with [B, 4, 4] x [B, 4, N]
    # [B, 4, N]
    pts_T_homog = torch.matmul(T.type_as(pts), pts_homog.transpose(1,2))
    pts_T = pts_T_homog.transpose(1,2)[:, :, :3]

    return pts_T


def uv_to_xy(uv, # torch.Tensor [M, 2] or [B, N, 2], trailing dimension is 2
             H, # image height
             W, # image width
             ): # torch Tensor with same shape as uv

    """
    Converts uv image coordinate to xy image coordinates.

    uv in [0, W] x [0, H]
    xy in [-1,1] x [-1,1]

    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """

    # x = 2.0*u/W  - 1
    x = uv.select(-1, 0)*2.0/W - 1.0
    y = uv.select(-1, 1)*2.0/H - 1.0
    xy = torch.stack((x,y), dim=-1)
    return xy



def get_freer_gpu():
    filename = "/tmp/gpu_stats.txt"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > %s' %(filename))
    memory_available = [int(x.split()[2]) for x in open(filename, 'r').readlines()]
    return np.argmax(memory_available)











