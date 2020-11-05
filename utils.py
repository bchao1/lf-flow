import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates

def normalize(img):
    """ Perform min-max normalization on image range """
    
    img = np.where(img == np.inf, 0, img)
    img = np.where(img == -np.inf, 0, img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def warp_image(img, flow_x, flow_y):
    h, w = img.shape[0], img.shape[1]
    x, y = np.meshgrid(np.arange(w), np.arange(h)) # original grids
    coordinates = np.stack([x.ravel(), y.ravel()]).T
    
    #mask = np.ones((h, w))
    warped_x = x + flow_x
    #mask[np.where(warped_x >= w)] = 0
    warped_y = y + flow_y
    warped_coodinates = np.stack([warped_x.ravel(), warped_y.ravel()]).T

    new_img = np.zeros((h, w, 3))
    for i in range(3):
        channel = img[:, :, i]
        new_img[:, :, i] = map_coordinates(channel, [warped_y, warped_x])
    new_img = new_img.astype(np.uint8)
    return new_img

def generate_lf(img, disp, lf_resolution):
    lf_x, lf_y = np.meshgrid(np.arange(lf_resolution), np.arange(lf_resolution))
    lf = np.zeros((lf_resolution, lf_resolution, *img.shape))

    lf_shifts = np.dstack([lf_y - lf_resolution // 2, lf_x - lf_resolution // 2])

    for i in range(lf_resolution):
        for j in range(lf_resolution):
            dy, dx = lf_shifts[i][j]
            view = warp_image(img, disp * dx, disp * dy)
            lf[i, j] = view
    return lf

def refocus(lf, pixels):
    from scipy import interpolate
    
    lf_h, lf_w, h, w, _ = lf.shape
    refocused_img = np.zeros((h, w, 3), dtype=np.float)
    X = np.arange(w).astype(np.float)
    Y = np.arange(h).astype(np.float)

    for ky in range(lf_h): # y
        for kx in range(lf_w): # x
            view = lf[ky, kx].astype(np.float)
            lf_y_shift = pixels * (ky - lf_h // 2)
            lf_x_shift = pixels * (kx - lf_w // 2)
            new_X = X + lf_x_shift
            new_Y = Y + lf_y_shift
            shifted_img = np.zeros((h, w, 3), dtype=np.float)
            for c in range(3):
                f = interpolate.interp2d(X, Y, view[:, :, c])
                shifted_img[:, :, c] = f(new_X, new_Y)
            refocused_img += shifted_img.astype(np.float)
    refocused_img /= (lf_h * lf_w)
    return (refocused_img * 255).astype(np.uint8)