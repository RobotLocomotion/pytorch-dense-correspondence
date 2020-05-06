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











