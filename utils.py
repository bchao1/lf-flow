import os
import numpy as np
import torch 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates


import torch
import torch.nn.functional as F

def normalize(img):
    """ Perform min-max normalization on image range """
    
    img = np.where(img == np.inf, 0, img)
    img = np.where(img == -np.inf, 0, img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def denorm_tanh(img):
    return (img + 1) * 0.5

def denorm_sigmoid(img):
    pass

def to_gray(img):
    assert img.shape[-1] == 3
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def warp_image_batch(img, flow_x, flow_y):
    """ Warp a 4D tensor 
        img: [N, C, H, W]
        flow_x: [N, H, W]
        flow_y: [N, H, W]
    """
    n, c, h, w = img.shape
    grid_h, grid_w = torch.meshgrid(torch.tensor(np.arange(h)), torch.tensor(np.arange(w)))
    grid_h = grid_h.cuda()
    grid_w = grid_w.cuda()
    grid = torch.stack([grid_w, grid_h], dim=-1).repeat(n, 1, 1, 1).cuda()
    flow = torch.stack([flow_x, flow_y], dim=-1)
    new_grid = grid + flow
    new_grid[:, :, :, 0] = 2 * new_grid[:, :, :, 0].clone() / (w - 1) - 1
    new_grid[:, :, :, 1] = 2 * new_grid[:, :, :, 1].clone() / (h - 1) - 1
    new_image = F.grid_sample(img, new_grid, align_corners=False)
    return new_image

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
    #new_img = new_img.astype(np.uint8)
    return new_img

def warp_strip(view, disp, low, high, mode='horizontal'):
    warped_imgs = []
    for i in range(low, high):
        if mode == 'horizontal':
            img = warp_image(view, disp * i, 0)
        elif mode == 'vertical':
            img = warp_image(view, 0, disp * i)
        else:
            raise Error()
        warped_imgs.append(img)
    return np.stack(warped_imgs)

def generate_lf_batch(img, col_idxs, disp, lf_resolution):
    n = img.shape[0]
    lf = []
    disp_x = []
    disp_y = []
    row_mid = lf_resolution // 2
    row_idxs = torch.tensor(row_mid).repeat(n).to(img.device) # middle row for stereo
    for i in range(lf_resolution): # row
        for j in range(lf_resolution): # col
            row_shear = (i - row_idxs).view(n, 1, 1).cuda()
            col_shear = (j - col_idxs).view(n, 1, 1).cuda()
            dy = disp * row_shear
            dx = disp * col_shear
            #disp_x.append(dx.unsqueeze(1))
            #disp_y.append(dy.unsqueeze(1))
            view = warp_image_batch(img, dx, dy)
            lf.append(view.unsqueeze(1))
    #disp_x = torch.cat(disp_x, dim=1).unsqueeze(2)
    #disp_y = torch.cat(disp_y, dim=1).unsqueeze(2)
    #flows = torch.cat([disp_x, disp_y], dim=2)
    lf = torch.cat(lf, dim=1)
    return lf#, flows

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

def animate_sequence(img_seq):
    """
        Animate an image sequence 
    """

    fig = plt.figure()
    im = plt.imshow(img_seq[0], cmap='gray')

    while True:
        for img in img_seq:
            im.set_data(img)
            fig.canvas.draw_idle()
            plt.pause(1)

def plot_loss_logs(log, name, save_dir):
    plt.figure()
    plt.plot(log)
    plt.title(name)
    plt.savefig(os.path.join(save_dir, "{}.png".format(name)))
    plt.close()

def save_lf_data(lf, epoch, save_dir):
    pass

if __name__ == '__main__':
    #lf = np.load("temp/lf.npy")
    #lf = lf.reshape(lf.shape[0] * lf.shape[1], *lf.shape[2:])
    #target = np.array(Image.open('./temp/target.png'))
    #syn_right = np.array(Image.open('./temp/syn_right.png'))
    #syn_left = np.array(Image.open('./temp/syn_left.png'))
    #err_left = np.array(Image.open('./temp/err_left.png'))
    #print(err_left)
    #err_right = np.array(Image.open('./temp/err_right.png'))
    #animate_sequence([err_left, err_right])
    lf = np.load("./experiments/syn_lf_100.npy")
    lf = denorm_tanh(lf)[0]
    lf = lf.reshape(9, 9, *lf.shape[1:])
    print(lf.shape)