import torch

def create_heatmap(uv_input, # [N, 2] tensor
                   H, # int: width
                   W, # int: height
                   sigma, # float: variance
                   type, # str ['softmax', 'exp']
                   ): # [N,H,W] tensor, each [n,:,:] is a heatmap

    N = uv_input.shape[0]
    # [N, H, W] with [:, :, w] = w
    u_grid = torch.arange(W).unsqueeze(0).unsqueeze(0).expand(*[N, H, -1])

    # [N, H, W] with [:, h, :] = h
    v_grid = torch.arange(H).unsqueeze(0).unsqueeze(-1).expand(*[N,-1, W])


    # [N, H, W, 2] in (u,v) ordering
    uv_grid = torch.stack((u_grid, v_grid), dim=-1).to(uv_input.device)

    # [N, H, W, 2]
    uv_input_grid = uv_input.unsqueeze(1).unsqueeze(1).expand(*[-1, H, W, -1])

    # print("uv_grid.shape", uv_grid.shape)
    # print("uv_input_grid.shape", uv_input_grid.shape)

    # [N, H, W]
    exp_arg = -1.0 * (uv_input_grid - uv_grid).type(torch.float).norm(p=2, dim=-1)/(sigma**2)

    # [N, H, W]
    heatmap = None
    if type == "exp":
        heatmap = torch.exp(exp_arg)
    elif type == "softmax":
        heatmap = torch.exp(exp_arg)
        heatmap = heatmap / (torch.sum(heatmap) + 1e-8)
    else:
        raise ValueError("unknown type: %s" %(type))

    return heatmap


def compute_heatmap_from_descriptors(des,  # [B, N, D]
                                     img,  # [B, D, H, W]
                                     sigma,  # float
                                     type,  # str ['exp', 'softmax']
                                     ): # [B, N, H, W] heatmaps

    # check dimensions
    assert des.shape[0] == img.shape[0]
    assert des.shape[2] == img.shape[1]

    N = des.shape[1]
    H = img.shape[2]
    W = img.shape[3]

    # [B, N, D, H, W]
    img_expand = img.unsqueeze(1).expand(*[-1, N, -1, -1, -1])

    # [B, N, H, W, D]
    # [B, N, D, H, W]
    des_expand = des.unsqueeze(-1).unsqueeze(-1).expand(*[-1,-1, -1, H, W])

    neg_squared_norm_diff = -1.0*(img_expand - des_expand).pow(2).sum(dim=2)*1.0/(sigma**2)

    # [B, N, H, W]
    heatmap = None
    if type == "exp":
        heatmap = torch.exp(neg_squared_norm_diff)
    elif type == "softmax":
        heatmap = torch.exp(neg_squared_norm_diff)
        heatmap = heatmap / (torch.sum(heatmap) + 1e-8)

    return heatmap





