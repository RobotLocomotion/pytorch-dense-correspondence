import torch

import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence.loss_functions.utils import compute_heatmap_from_descriptors

def get_integral_preds_2d(heatmaps, # [N, H, W]
                          verbose=False,
                          ): # [N, 2] uv coordinates ordering

    device = heatmaps.device
    N, H, W = heatmaps.shape

    # normalize the heatmaps so that they sum to one
    # why are we doing keep_dim=True . . so that broadcasting works easier
    heatmaps_sum = torch.sum(heatmaps, dim=[1,2], keepdim=True)

    # this heatmap is now of shape
    # [N, H, W] but each heatmaps_norm[i] which is W x H sums to one
    # heatmaps_norm = heatmaps/heatmaps_sum

    # [N,H,W,2]
    heatmaps_norm = (heatmaps/(heatmaps_sum + 1e-8)).unsqueeze(-1).expand(*[-1,-1,-1, 2])

    if verbose:
        print("heatmaps_norm.shape", heatmaps_norm.shape)

    # [N, H, W] with [:, :, w] = w
    u_grid = torch.arange(W).unsqueeze(0).unsqueeze(0).expand(*[N, H, -1])

    # [N, H, W] with [:, h, :] = h
    v_grid = torch.arange(H).unsqueeze(0).unsqueeze(-1).expand(*[N, -1, W])

    # [N, H, W, 2] in (u,v) ordering
    uv_grid = torch.stack((u_grid, v_grid), dim=-1).to(device).type(torch.float)

    if verbose:
        print("uv_grid.shape", uv_grid.shape)

    # [N, 2]
    pred = torch.sum((heatmaps_norm * uv_grid), dim=[1,2])

    return pred

def get_integral_preds_3d(heatmaps, # [N, H, W]
                          depth_images, # [N, H, W], these are kept in the same units they came in
                          compute_uv=True,
                          verbose=False,
                          ): # dict containing two tensors {'uv': [N, 2], 'z': [N,]}
    """

    Computes spatial expectation to get uv pixel location.
    Uses spatial expectation together with a "fixed" depth image to predict
    a depth value for each of the N input heatmaps
    """

    device = heatmaps.device
    N, H, W = heatmaps.shape

    assert heatmaps.shape == depth_images.shape, "dimension mismatch"
    # N, H_2, W_2 = depth_images.shape
    # assert (H == H_2) and (W_2 == W), "dimension mismatch"

    # normalize the heatmaps so that they sum to one
    # why are we doing keep_dim=True . . so that broadcasting works easier
    # [N,]
    heatmaps_sum = torch.sum(heatmaps, dim=[1,2], keepdim=True)

    # this heatmap is now of shape
    # [N, H, W] but each heatmaps_norm[i] which is W x H sums to one
    # heatmaps_norm = heatmaps/heatmaps_sum


    pred = None
    if compute_uv:
        # [N,H,W,2]
        heatmaps_norm = (heatmaps/(heatmaps_sum + 1e-8)).unsqueeze(-1).expand(*[-1,-1,-1, 2])

        if verbose:
            print("heatmaps_norm.shape", heatmaps_norm.shape)

        # [N, H, W] with [:, :, w] = w
        u_grid = torch.arange(W).unsqueeze(0).unsqueeze(0).expand(*[N, H, -1])

        # [N, H, W] with [:, h, :] = h
        v_grid = torch.arange(H).unsqueeze(0).unsqueeze(-1).expand(*[N, -1, W])

        # [N, H, W, 2] in (u,v) ordering
        uv_grid = torch.stack((u_grid, v_grid), dim=-1).to(device).type(torch.float)

        if verbose:
            print("uv_grid.shape", uv_grid.shape)

        # [N, 2]
        # u,v tensor
        pred = torch.sum((heatmaps_norm * uv_grid), dim=[1,2])

    # only use valid depth values (i.e. those that are > 0)
    # renormalize the heatmap to sum to one over those value
    # [N, H, W], dtype=bool
    depth_valid_mask = (depth_images > 0)

    # normalization factor
    # [N,]
    heatmaps_valid_norm = (depth_valid_mask.float() * heatmaps).sum(dim=[1,2])

    # [N,]
    z_pred = (depth_valid_mask.float() * heatmaps * depth_images).sum(dim=[1,2]) / (heatmaps_valid_norm + 1e-8)

    return {'uv': pred,
            'z': z_pred}


def get_argmax_l2(des, # [B, N, D] or [N,D]
                  des_img, # [B, D, H, W]
                  ):

    B, D, H, W = des_img.shape

    # [B, N, D]
    des_unsqueeze = None
    if len(des.shape) == 2: #[N, D] case
        des_unsqueeze = des.unsqueeze(0).expand(*[B, -1, -1])
    elif len(des.shape) == 3:
        des_unsqueeze = des
    else:
        raise ValueError("dimension mismatch")

    B2, N, D2 = des_unsqueeze.shape
    assert (B2 == B) and (D2 == D), "dimension mismatch"

    # [B, N, D, H, W]
    expand_batch_des_a = pdc_utils.expand_descriptor_batch(des_unsqueeze, H, W)

    # [B, N, D, H, W]
    expand_des_img_b = pdc_utils.expand_image_batch(des_img, N)

    norm_diff = (expand_batch_des_a - expand_des_img_b).norm(p=2, dim=2)

    best_match_dict = pdc_utils.find_pixelwise_extreme(norm_diff, type="min")

    return best_match_dict

def get_spatial_expectation(des, # [B, N, D] or [N, D]
                            des_img, # [B, D, H, W] of [D, H, W]
                            sigma, # float
                            type, # str ['exp', 'softmax']
                            return_heatmap=False,
                            compute_heatmap_values=False,
                            ): # [B, N, 2] or [N, 2] in uv ordering
    """
    Computes the spatial expectation in 2D

    Essentially this function combines two other functions.

    1) Create the heatmaps
    2) Perform the integral operation

    """


    has_batch_dim = (len(des_img.shape) == 4)

    if not has_batch_dim:
        des_img = des_img.unsqueeze(0)

    B, D, H, W = des_img.shape

    # [B, N, D]
    if len(des.shape) == 2:  # [N, D] case
        des = des.unsqueeze(0).expand(*[B, -1, -1])
    elif len(des.shape) != 3:
        raise ValueError("dimension mismatch")

    B2, N, D2 = des.shape
    assert (B2 == B) and (D2 == D), "dimension mismatch"


    # [B, N, H, W]
    heatmap = compute_heatmap_from_descriptors(des, des_img, sigma, type)

    # collapse to [M, H, W] where M = B*N
    heatmap_no_batch = heatmap.reshape([B*N, H, W])

    # [M, 2]
    uv = get_integral_preds_2d(heatmap_no_batch)
    M, _ = uv.shape

    # index into the batch image tensor
    heatmap_values = None
    if compute_heatmap_values:
        # [M, 2]
        uv_int = uv.type(torch.LongTensor).to(uv.device) # cast it to

        # index into heatmap_no_batch
        first_idx = torch.arange(M)

        # [M,]
        heatmap_values = heatmap_no_batch[first_idx, uv_int[:, 1], uv_int[:, 0]]
        print("heatmap_values.shape", heatmap_values.shape)

    # reshape to original dimensions
    if has_batch_dim:
        uv = uv.reshape([B, N, 2])

        if heatmap_values is not None:
            heatmap_values = heatmap_values.reshape([B, N])

    if return_heatmap:
        return {'heatmap': heatmap,
                'heatmap_no_batch': heatmap_no_batch,
                'uv': uv,
                'heatmap_values': heatmap_values,
                }
    else:
        return {'uv': uv,
                'heatmap_values': heatmap_values}










