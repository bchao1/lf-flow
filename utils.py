import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
from collections import Counter


import torch
import torch.nn.functional as F

class Dummy:
    def __init__(self):
        pass
        
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
    grid_h = grid_h.to(img.device)
    grid_w = grid_w.to(img.device)
    grid = torch.stack([grid_w, grid_h], dim=-1).repeat(n, 1, 1, 1).to(img.device)
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

def generate_lf_batch(img, row_idxs, col_idxs, disp, lf_resolution, scale=1):
    n = img.shape[0]
    lf = []
    disp_x = []
    disp_y = []
    for i in range(lf_resolution): # row
        for j in range(lf_resolution): # col
            row_shear = (i - row_idxs).view(n, 1, 1).to(img.device)
            col_shear = (j - col_idxs).view(n, 1, 1).to(img.device)
            dy = disp * row_shear * scale
            dx = disp * col_shear * scale
            view = warp_image_batch(img, dx, dy)
            lf.append(view.unsqueeze(1))
    lf = torch.cat(lf, dim=1)
    return lf#, flows

def generate_lf_batch_single_image(img, depths, lf_res):
    lf = []
    for i in range(lf_res):
        for j in range(lf_res):
            depth = depths[:, i * lf_res + j]
            row_shear = i - lf_res // 2
            col_shear = j - lf_res // 2
            dy = depth * row_shear
            dx = depth * col_shear
            view = warp_image_batch(img, dx, dy)
            lf.append(view.unsqueeze(1))
    lf = torch.cat(lf, dim=1)
    return lf

def shift_depths(depths, delX, delY):
    b, n, h, w = depths.shape
    dx_unit = torch.empty(n, h, w).fill_(delX).to(depths.device)
    dy_unit = torch.empty(n, h, w).fill_(delY).to(depths.device)
    shifted_depths = []
    for i in range(b):
        depth_lf = depths[i].unsqueeze(1)
        dx = dx_unit * depths[i]
        dy = dy_unit * depths[i]
        shifted_depth = warp_image_batch(depth_lf, dx, dy).squeeze().unsqueeze(0)
        shifted_depths.append(shifted_depth)
    return torch.cat(shifted_depths, dim=0)



def compute_alpha_blending(row_idx, left_idx, right_idx, left_lf, right_lf, lf_res):
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

    # light field positions
    x_dist = torch.arange(lf_res).repeat(lf_res, 1).float().to(device)
    y_dist = x_dist.T
    x_dist = x_dist.unsqueeze(0).repeat(n, 1, 1)
    y_dist = y_dist.unsqueeze(0).repeat(n, 1, 1)
    
    l_shift = x_dist - left_idx.view(n, 1, 1)
    r_shift = x_dist - right_idx.view(n, 1, 1)
    y_shift = y_dist - row_idx.view(n, 1, 1)

    left_dist = torch.sqrt(torch.pow(l_shift, 2) + torch.pow(y_shift, 2))
    right_dist = torch.sqrt(torch.pow(r_shift, 2) + torch.pow(y_shift, 2))
    weights = left_dist + right_dist
    # linear blending
    alpha = right_dist / weights # alpha value for left view
    alpha = alpha.view(n, num_views, 1, 1, 1)
    blended_lf = alpha * left_lf + (1 - alpha) * right_lf
    return blended_lf

def get_weight_map(row_idx, left_idx, right_idx, lf_res, reduce=True):
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
    if reduce: # scale weights
        w_sum = torch.sum(avg_dist.view(n, -1), dim=1).view(n, 1, 1)
        avg_dist = (avg_dist / w_sum) * (lf_res**2) # scale weights
        avg_dist = avg_dist.view(n, -1, 1, 1, 1)
    return avg_dist
    
def view_loss_to_dist(x, target, row_idx, left_idx, right_idx, lf_res, loss_func):
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
    n, num_views, _, h, w = x.shape
    dist = get_weight_map(row_idx, left_idx, right_idx, lf_res, reduce=False) # (N, lf_res, lf_res)
    dist = dist.view(n, num_views) # (N, num_views)
    loss = loss_func(x, target, reduction="none")
    loss = torch.mean(loss.view(n, num_views, -1), dim=-1) # (N, num_views)

    dist = dist.view(-1).detach().cpu().numpy() # flatten
    loss = loss.view(-1).detach().cpu().numpy() # flatten

    dist_loss = dict.fromkeys(dist, 0)
    dist_cnt = dict.fromkeys(dist, 0)
    
    for d, l in zip(dist, loss):
        dist_loss[d] += l
        dist_cnt[d] += 1
    for d in dist_loss.keys():
        dist_loss[d] /= dist_cnt[d]
    dist = sorted(dist_loss.keys())
    loss = [dist_loss[d] for d in dist]
    return np.array(dist), np.array(loss)



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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def animate_sequence(img_seq, pause=1):
    """
        Animate an image sequence 
    """

    fig = plt.figure()
    im = plt.imshow(img_seq[0], cmap='gray')

    while True:
        for i, img in enumerate(img_seq):
            im.set_data(img)
            fig.canvas.draw_idle()
            #plt.title(i)
            plt.pause(pause)

def plot_loss_logs(log, name, save_dir):
    plt.figure()
    plt.plot(log)
    plt.title(name)
    plt.savefig(os.path.join(save_dir, "{}.png".format(name)))
    plt.close()

def save_lf_data(lf, epoch, save_dir):
    pass

def compute_focal_stack(lf, f):
    refocused_images = [refocus(lf, p) for p in f]
    focal_stack = np.stack(refocused_images)
    return focal_stack

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

def test_view_loss_against_dist():
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
    x = torch.randn(b, lf_res**2, 3, h, w)
    target = torch.randn(b, lf_res**2, 3, h, w)
    view_loss_to_dist(x, target, row_idx, left_idx, right_idx, lf_res, F.l1_loss)

def test_left_right_error():
    """
    imgs = []
    k = 6
    for i in range(7):
        img = Image.open("./temp/flows/disp_{}_{}.png".format(k, i))
        img = np.array(img)
        imgs.append(img)
    animate_sequence(imgs)
    """
    from image_utils import lf_to_multiview
    def read_lf(path):
        lf = np.load(path.format(i))[0]
        lf = np.transpose(lf, (0, 2, 3 ,1))
        return lf

    i = 1
    merged_lf = read_lf("./temp/merged_{}.npy".format(i))
    target_lf = read_lf("./temp/target_{}.npy".format(i))
    left_lf = read_lf("./temp/left_{}.npy".format(i))
    right_lf = read_lf("./temp/right_{}.npy".format(i))

    def get_err_map(lf, target_lf):
        num_views, h, w, _ = lf.shape
        lf_res = int(np.sqrt(num_views))
        err = (lf - target_lf)**2
        err = np.mean(err, axis=-1)
        err = (err - np.min(err, axis=(-1, -2), keepdims=True))/(np.max(err, axis=(-1, -2), keepdims=True) - np.min(err, axis=(-1, -2), keepdims=True))
        err = err.reshape(lf_res, lf_res, h, w)
        err = np.transpose(err, (0, 2, 1, 3))
        err = err.reshape(lf_res*h, lf_res*w)
        err = (err * 255).astype(np.uint8)
        print(err.shape)
        return err

    #Image.fromarray(get_err_map(left_lf, target_lf)).save("./temp/left_err.png")
    #Image.fromarray(get_err_map(right_lf, target_lf)).save("./temp/right_err.png")
    #Image.fromarray(get_err_map(merged_lf, target_lf)).save("./temp/merged_err.png")
    
    n, h, w, c = merged_lf.shape
    l = int(np.sqrt(n))
    merged_lf = merged_lf.reshape(l, l, h, w,c) # 0,1, 2, 3, 4
    merged_lf = np.transpose(merged_lf, (0, 2, 1, 3, 4))
    merged_lf = merged_lf.reshape(l*h, l*w, c)
    mv = merged_lf
    #mv = lf_to_multiview(merged_lf)
    mv = (mv*255).astype(np.uint8)
    Image.fromarray(mv).save('./temp/test.png')
    #animate_sequence(merged_lf)

def test_focal_stack():
    from image_utils import save_image
    lf = np.load("./temp/bunny_focal_stack/syn.npy")
    u, v, h, w, c = lf.shape
    #print(lf.shape)
    #lf = lf.reshape(u*v, h, w, 3)
    #animate_sequence(lf, 0.1)
    #exit()
    f = np.linspace(-0.5, 0.5, 20)
    front = refocus(lf, f[0])
    middle = refocus(lf, f[10])
    back = refocus(lf, f[-1])

    def get_patch(img, y_lo, y_hi, x_lo, x_hi):
        return img[y_lo:y_hi, x_lo: x_hi, :]
    front_bunny_pos = [200, 300, 100, 200]
    middle_bunny_pos = [100, 200, 50, 150]
    backdrop_pos = [0, 100, 50, 150]

    front_bunny_1 = get_patch(front, *front_bunny_pos)
    front_bunny_2 = get_patch(middle, *front_bunny_pos)
    front_bunny_3 = get_patch(back, *front_bunny_pos)

    middle_bunny_1 = get_patch(front, *middle_bunny_pos)
    middle_bunny_2 = get_patch(middle, *middle_bunny_pos)
    middle_bunny_3 = get_patch(back, *middle_bunny_pos)

    backdrop_1 = get_patch(front, *backdrop_pos)
    backdrop_2 = get_patch(middle, *backdrop_pos)
    backdrop_3 = get_patch(back, *backdrop_pos)
    

    #
    save_image(front_bunny_1, "./data/front_1.png")
    save_image(front_bunny_2, "./data/front_2.png")
    save_image(front_bunny_3, "./data/front_3.png")

    save_image(middle_bunny_1, "./data/middle_1.png")
    save_image(middle_bunny_2, "./data/middle_2.png")
    save_image(middle_bunny_3, "./data/middle_3.png")

    save_image(backdrop_1, "./data/backdrop_1.png")
    save_image(backdrop_2, "./data/backdrop_2.png")
    save_image(backdrop_3, "./data/backdrop_3.png")

def test_left_right_heat_map():
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.style.use('ggplot')
    err_left = np.load("./temp/err_left_1.npy").reshape(9, 9)
    err_right = np.load("./temp/err_right_1.npy").reshape(9, 9)
    err_merged = np.load("./temp/err_merged_1.npy").reshape(9, 9)
    err_maps = [err_left, err_right, err_merged]
    name = ['left-synthesized', 'right-synthesized', 'alpha-blended']
    min_val = np.min(np.concatenate(err_maps).ravel())
    max_val = np.max(np.concatenate(err_maps).ravel())
    
    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        im = axs[i].imshow(err_maps[i], vmin=min_val, vmax=max_val, cmap='jet')
        axs[i].grid(False)
        axs[i].tick_params(axis='both', which='major', labelsize=5)
        axs[i].set_xticks(list(range(9)))
        axs[i].set_yticks(list(range(9)))
        axs[i].set_title("Mean error of sub-aperture views \n from {} light field".format(name[i]), size=5)
    #divider = make_axes_locatable(axs[-1])
    plt.colorbar(im, ax=axs)
    plt.show()

def normalize(z):
    return (z - np.min(z.ravel())) / (np.max(z.ravel()) - np.min(z.ravel()))

if __name__ == '__main__':
    i = 9
    dataset = "inria"
    comp_1 = "./temp/{}_{}/".format(dataset, "srinivasan")
    comp_2 = "./temp/{}_{}/".format(dataset, "kalantari")
    ours = "./temp/{}_{}/".format(dataset, "ours")
    
    image = "tl_syn_{}.png".format(i)
    target = "tl_target_{}.png".format(i)

    img1 = np.array(Image.open(os.path.join(comp_1, image)))
    img2 = np.array(Image.open(os.path.join(comp_2, image)))
    img3 = np.array(Image.open(os.path.join(ours, image)))
    t = np.array(Image.open(os.path.join(ours, target)))

    animate_sequence([img2, t])
