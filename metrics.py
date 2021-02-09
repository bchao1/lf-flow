import skimage.metrics as metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import shift_depths

def psnr(img, target):
    return metrics.peak_signal_noise_ratio(target, img)

def mse(img, target):
    return metrics.mean_squared_error(img, target)

def ssim(img, target, mode=0):
    if mode == 0:
        img, target = img.squeeze(), target.squeeze() # reduce batch dimension
        img = np.transpose(img, (0, 2, 3, 1))
        target = np.transpose(target, (0, 2, 3, 1))
        ssim_sum = 0
        for i in range(len(img)):
            ssim_sum += metrics.structural_similarity(target[0], img[0], 
                data_range=img[0].max() - img[0].min(), multichannel=True)
        return ssim_sum / len(img)
    else:
        img = np.transpose(img, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
        return metrics.structural_similarity(target, img, 
                data_range=img.max() - img.min(), multichannel=True)

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

class TVLoss(nn.Module):
    """ Regularization to constrain the total variation of the disparity / depth map """
    def __init__(self, w):
        super(TVLoss, self).__init__()
        self.w = w
    
    def __call__(self, img):
        # img: (b, 1, H, W) disparity map
        # img: (b, N, H, W) # is num views
        # Compute x-gradient and y-gradient. Minimize variation
        b, n, h, w = img.shape
        img = img.contiguous().view(b*n, 1, h, w)

        x_kernel = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]).float().to(img.device).view(1, 1, 3, 3)

        y_kernel = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]).float().to(img.device).view(1, 1, 3, 3)

        G_x = F.conv2d(img, x_kernel, padding=1).squeeze()
        G_y = F.conv2d(img, y_kernel, padding=1).squeeze()
        loss = torch.mean(torch.abs(G_x) + torch.abs(G_y))
        return loss * self.w

class DepthConsistencyLoss(nn.Module):
    def __init__(self, w):
        super(DepthConsistencyLoss, self).__init__()
        self.w = w
    
    def __call__(self, depths):
        b, n, h, w = depths.shape
        lf_res = int(np.sqrt(n))
        shiftedX = shift_depths(depths, 1, 0).reshape(b, lf_res, lf_res, h, w)
        shiftedY = shift_depths(depths, 0, 1).reshape(b, lf_res, lf_res, h, w)
        shiftedXY = shift_depths(depths, 1, 1).reshape(b, lf_res, lf_res, h, w)
        depths = depths.reshape(b, lf_res, lf_res, h, w)
        
        l1 = torch.abs(shiftedX[:, 1:, 1:, :, :] - depths[:, 1:, 1:, :, :])
        l2 = torch.abs(shiftedY[:, 1:, 1:, :, :] - depths[:, 1:, 1:, :, :])
        l3 = torch.abs(shiftedXY[:, 1:, 1:, :, :] - depths[:, 1:, 1:, :, :])
        loss = torch.mean(l1 + l2 + l3)
        return loss * self.w


class WeightedReconstructionLoss(nn.Module):
    def __init__(self, loss_func):
        super(WeightedReconstructionLoss, self).__init__()
        self.loss_func = loss_func 
    
    def __call__(self, x, target, w = 1):
        assert x.shape == target.shape
        if torch.is_tensor(w):
            w = w.expand_as(x) # avoid broadcasting and large memory overhead
        return self.loss_func(x * w, target * w)