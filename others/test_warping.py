import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch.nn.functional import grid_sample

def test():
    h = 64
    w = 64
    x = torch.randn(1, h, w) # to warp
    flow_h = torch.randn(h, w) * 5
    flow_w = torch.randn(h, w) * 5
    h_list = torch.tensor(np.arange(h))
    w_list = torch.tensor(np.arange(w))
    grid_h, grid_w = torch.meshgrid(h_list, w_list)
    
    new_h = grid_h + flow_h
    new_w = grid_w + flow_w

    # 
    new_h = 2 * new_h / h - 1
    new_w = 2 * new_w / w - 1
    grid_map = torch.stack([new_h, new_w], dim=-1).unsqueeze(0)
    warp_torch = grid_sample(x.unsqueeze(0), grid_map, align_corners=False).squeeze().numpy()

    warp_scipy = map_coordinates(x[0].numpy(), [new_h.numpy(), new_w.numpy()])

    print("grid sample:")
    print(warp_torch)

    print("map coordinate:")
    print(warp_scipy)

if __name__ == '__main__':
    test()