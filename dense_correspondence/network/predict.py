import torch
import dense_correspondence_manipulation.utils.utils as pdc_utils

def get_integral_preds_2d(heatmaps, # [N, H, W]
                          verbose=False,
                          ): # [N, 2] uv coordinates

    device = heatmaps.device
    N, H, W = heatmaps.shape

    # normalize the heatmaps so that they sum to one
    # [N, 1, 1]
    heatmaps_sum = torch.sum(heatmaps, dim=[1,2], keepdim=True)
    heatmaps_norm = heatmaps/heatmaps_sum

    # [N,H,W,2]
    heatmaps_norm = heatmaps_norm.unsqueeze(-1).expand(*[-1,-1,-1, 2])

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

def get_argmax_l2(des, # [B, N, D] or [N,D]
                  des_img, # [B, D, H, W]
                  ):

    B, D, H, W = des_img.shape
    N = des.shape[0]

    # [B, N, D]
    des_unsqueeze = None
    if len(des.shape) == 2:
        des_unsqueeze = des.unsqueeze(0).expand(*[B, -1, -1])
    elif len(des.shape) == 3:
        B2, _, D2 = des.shape
        assert B2 == B
        assert D2 == D
        des_unsqueeze = des
    else:
        raise ValueError("dimension mismatch")

    # [B, N, D, H, W]
    expand_batch_des_a = pdc_utils.expand_descriptor_batch(des_unsqueeze, H, W)

    # [B, N, D, H, W]
    expand_des_img_b = pdc_utils.expand_image_batch(des_img, N)

    # print("expand_batch_des_a.shape", expand_batch_des_a.shape)
    # print("expand_des_img_b.shape", expand_des_img_b.shape)

    # [B, N, H, W]
    norm_diff = (expand_batch_des_a - expand_des_img_b).norm(p=2, dim=2)

    best_match_dict = pdc_utils.find_pixelwise_extreme(norm_diff, type="min")

    return best_match_dict



