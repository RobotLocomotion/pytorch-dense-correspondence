import torch

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

def get_argmax_l2(des, # [N, D]
                  des_img, # [N, D, H, W]
                  ):

    N, D, H, W = des_img.shape

    # [B, N, D, H, W]
    expand_batch_des_a = pdc_utils.expand_descriptor_batch(des.unsqueeze(0), H, W)
    expand_des_img_b = pdc_utils.expand_image_batch(des_img.unsqueeze(0), N)

    # [B, N, H, W]
    norm_diff = (expand_batch_des_a - expand_des_img_b).norm(p=2, dim=2)

    best_match_dict = pdc_utils.find_pixelwise_extreme(norm_diff, type="min")

    best_match_dict['best_match_dict']['indices'].squeeze(0)



