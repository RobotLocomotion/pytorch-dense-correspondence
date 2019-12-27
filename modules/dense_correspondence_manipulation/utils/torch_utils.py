from torchvision import transforms
import torch

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









