import os
import numpy as np
import torch 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates


import torch
import torch.nn.functional as F

class AverageMeter:
    def __init__(self):
        self._n = 0
        self._sum = 0
        self._avg = 0
    
    def update(self, val, n = 1):
        self._n += n
        self._sum += val * n
        self._avg = self._sum / self._n

    @property
    def avg(self):
        return self._avg
    
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
    new_grid = torch.clamp(new_grid, -1, 1)
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

def generate_lf_batch(img, row_idxs, col_idxs, disp, lf_resolution):
    n = img.shape[0]
    lf = []
    disp_x = []
    disp_y = []
    for i in range(lf_resolution): # row
        for j in range(lf_resolution): # col
            row_shear = (i - row_idxs).view(n, 1, 1).cuda()
            col_shear = (j - col_idxs).view(n, 1, 1).cuda()
            dy = disp * row_shear
            dx = disp * col_shear
            view = warp_image_batch(img, dx, dy)
            lf.append(view.unsqueeze(1))
    lf = torch.cat(lf, dim=1)
    return lf#, flows

def compute_alpha_blending(left_idx, right_idx, left_lf, right_lf, lf_res):
    """
        Args:
            left_idx: (N, )
            right_idx: (N, )
            left_lf: (N, num_views, 3, h, w)
            right_lf: (N, num_views, 3, h, w)
            lf_res: (,)
        Returns:
            blended light field (N, num_views, 3, h, w)
    """
    n, num_views, _, _, _ = left_lf.shape
    assert num_views == lf_res**2
    device = left_lf.device

    x_dist = torch.arange(lf_res).repeat(lf_res, 1).float().to(device)
    x_dist = x_dist.unsqueeze(0).repeat(n, 1, 1)
    
    l_shift = torch.abs(x_dist - left_idx.view(n, 1, 1))
    r_shift = torch.abs(x_dist - right_idx.view(n, 1, 1))
    weights = l_shift + r_shift
    #print(weights[:, :, 0])
    alpha = r_shift / weights # alpha value for left view
    alpha = alpha.view(n, num_views, 1, 1, 1)
    blended_lf = alpha * left_lf + (1 - alpha) * right_lf
    return blended_lf

def get_weight_map(row_idx, left_idx, right_idx, lf_res):
    """
        Args:
            row_idx, left_idx, right_idx: (N, )
            lf_res: (,)
        Returns:
            Weighted loss map: (N, lf_res**2, 1, 1, 1). To multiply with light field
    """
    device = row_idx.device
    n = row_idx.shape[0]
    x_dist = torch.arange(lf_res).repeat(lf_res, 1).float()
    y_dist = x_dist.clone().T
    x_dist = x_dist.unsqueeze(0).repeat(n, 1, 1).to(device)
    y_dist = y_dist.unsqueeze(0).repeat(n, 1, 1).to(device)

    left_shift = torch.pow(x_dist - left_idx.view(n, 1, 1), 2)
    right_shift = torch.pow(x_dist - right_idx.view(n, 1, 1), 2)
    row_shift = torch.pow(y_dist - row_idx.view(n, 1, 1), 2)
    left_dist = torch.sqrt(left_shift + row_shift)
    right_dist = torch.sqrt(right_shift + row_shift)
    avg_dist = (left_dist + right_dist) * 0.5
    w_sum = torch.sum(avg_dist.view(n, -1), dim=1).view(n, 1, 1)
    avg_dist = (avg_dist / w_sum) * (lf_res**2) # scale weights
    avg_dist = avg_dist.view(n, -1, 1, 1, 1)
    return avg_dist
    
def compute_view_wise_loss(x, target, row_idx, left_idx, right_idx, loss_func):
    """
        Compute loss agaist distance of novel view to input view

        Args:
            x: (N, num_views, 3, H, W)
            target: (N, num_views, 3, H, W)
            row_idx: (N, )
            left_idx: (N, )
            right_idx: (N, )
            loss_func: to compute view-wise loss
    """

    assert x.shape == target.shape

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


# tests
def test_alpha_belding():
    from image_utils import sample_stereo_index
    b = 7
    lf_res = 9
    left_idx = []
    right_idx = []
    h, w = 128, 128
    for _ in range(b):
        _, l, r = sample_stereo_index(lf_res)
        left_idx.append(l)
        right_idx.append(r)
    left_idx = torch.tensor(left_idx)
    right_idx = torch.tensor(right_idx)
    left_lf = torch.randn(b, lf_res**2, 3, h, w)
    right_lf = torch.randn(b, lf_res**2, 3, h, w)
    lf = compute_alpha_blending(left_idx, right_idx, left_lf, right_lf, lf_res)
    print(lf.shape)
    
def test_weight_map():
    from image_utils import sample_stereo_index
    b = 7
    lf_res = 9
    left_idx = []
    right_idx = []
    row_idx = []
    h, w = 128, 128
    for _ in range(b):
        row, l, r = sample_stereo_index(lf_res)
        left_idx.append(l)
        right_idx.append(r)
        row_idx.append(row)
    left_idx = torch.tensor(left_idx)
    right_idx = torch.tensor(right_idx)
    row_idx = torch.tensor(row_idx)
    w = get_weight_map(row_idx, left_idx, right_idx, lf_res)

if __name__ == '__main__':
    #test_alpha_belding()
    test_weight_map()