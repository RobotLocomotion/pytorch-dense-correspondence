import torch

def create_heatmap(uv_input, # [N, 2] tensor
                   H, # int: width
                   W, # int: height
                   sigma, # float: variance
                   ): # [N,H,W] tensor each [n,:,:] is a heatmap

    N = uv_input.shape[0]
    # [N, H, W] with [:, :, w] = w
    u_grid = torch.arange(W).unsqueeze(0).unsqueeze(0).expand(*[N, H, -1])

    # [N, H, W] with [:, h, :] = h
    v_grid = torch.arange(H).unsqueeze(0).unsqueeze(-1).expand(*[N,-1, W])

    # [N, H, W, 2] in (u,v) ordering
    uv_grid = torch.stack((u_grid, v_grid), dim=-1)

    # [N, H, W, 2]
    uv_input_grid = uv_input.unsqueeze(1).unsqueeze(1).expand(*[-1, H, W, -1])

    print("uv_grid.shape", uv_grid.shape)
    print("uv_input_grid.shape", uv_input_grid.shape)

    # [N, H, W]
    exp_arg = -1.0 * (uv_input_grid - uv_grid).type(torch.float).norm(p=2, dim=-1)/(sigma**2)

    # [N, H, W]
    heatmap = torch.exp(exp_arg)


    return heatmap

