import skimage.metrics as metrics
import numpy as np
import torch

def psnr(img, target):
    return metrics.peak_signal_noise_ratio(target, img)

def mse(img, target):
    return metrics.mean_squared_error(img, target)

class ColorConstancyLoss:
    """ Compute the color constancy of a light field patch 

        Input size:
            Shape (N, lf*lf, 3, h, w) light field tensor
    """
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def __call__(self, x):
        n, num_views, c, h, w = x.shape
        # x: (N, lf*lf, 3, h, w)
        
        # Crop x to (N, lf*lf, 3, patch_size, patch_size)

        # Compute mean rgb color of each patch -> (N, lf*lf, 3)
        top_left_i = np.random.randint(low = 0, high = h - self.patch_size + 1) # [0, h - sz]
        top_left_j = np.random.randint(low = 0, high = w - self.patch_size + 1) # [0, w - sz]
        patch = x[:, :, :, top_left_i: top_left_i + self.patch_size, top_left_j: top_left_j + self.patch_size]
        patch = patch.contiguous().view(n, num_views, c, -1)
        patch_color_mean = torch.mean(patch, dim=-1) # (N, lf*lf, 3)
        color_std = torch.std(patch_color_mean, dim=1) # (N, 3) standard deviations
        loss = torch.mean(color_std.view(-1))
        return loss


