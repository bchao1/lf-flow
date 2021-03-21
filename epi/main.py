import sys 
sys.path.append("..")

import cv2
import numpy as np 
import matplotlib.pyplot as plt
from utils import animate_sequence
import metrics
from models.epinet import EPIVolumeNet

import torch
import torch.nn.functional as F
import logging

def angular_downsample_lf(lf, ratio):
    u, v, h, w, _ = lf.shape
    assert u == v # constrain to square light fields
    assert (u - 1) % ratio == 0
    return lf[::ratio, ::ratio, :, :, :]

def get_horizontal_strip(lf, row):
    assert 0 <= row < lf.shape[0]
    return lf[row, :]

def get_vertical_strip(lf, col):
    assert 0 <= col < lf.shape[1]
    return lf[:, col]

def get_vertical_epi_slice(lf, idx):
    assert 0 <= idx < lf.shape[2]
    return lf[:, :, idx, :]

def get_horizontal_epi_slice(lf, idx):
    assert 0 <= idx < lf.shape[1]
    return lf[:, idx, :, :]

def sample_indices(syn_combs, n_dense):
    """
        Sample indices for self-supervised light field view learning.
        Args:
            n_dense: Dense view dimensions.s
    """
    n_sparse, n_inter = syn_combs[np.random.randint(len(syn_combs))]
    n_syn = n_sparse + (n_sparse - 1) * n_inter
    start_index = np.random.randint(n_dense - n_syn + 1)
    end_index = start_index + n_syn - 1
    
    # all views = [start_index, start_index + n_syn)
    syn_indices = np.arange(start_index, start_index + n_syn)[::(n_inter + 1)]
    assert len(syn_indices) == n_sparse
    return syn_indices


def find_syn_combinations(n_input, n_dense):

    """
        a + (a-1) * b <= k
        1. a >= 3 
        2. b >= 1
            k >= a + (a - 1) * b >= 2a - 1
            a <= (k + 1) / 2
        Range of a: 
            3 <= a <= (k + 1) / 2

        Fixed a:
        1. b >= 1
        2. b <= (k - a) / (a - 1)
    """
    combs = []
    for n_sparse in range(3, (n_dense + 1) // 2 + 1):
        for n_inter in range(1, (n_dense - n_sparse) // (n_sparse - 1) + 1):
            if (n_sparse == n_input) and (n_sparse + (n_sparse - 1) * n_inter == n_dense):
                continue
            combs.append((n_sparse, n_inter))
    return combs

def ssl_train_views(lf, syn_combs, net):
    bsz, lf_res, _, _, h, w = lf.shape # (bsz, lf_res, lf_res, 3, h, w)

    # Training
    row_idx = np.random.randint(lf_res) # sample row / column view index
    sparse_row_views = lf[:, row_idx, :, :, :, :] # (bsz, N', 3, H, W)
    sparse_row_views = torch.transpose(sparse_row_views, 1, 2) # permute channel with view dimension
    # row views (bsz, 3, N', H, W) -> now upsample 3rd dimension N' -> N
    dense_row_views = F.interpolate(sparse_row_views, (lf_dense_res, h, w)) # upsample angular dimension
    # upsampled row views (bsz, 3, N, H, W)
    # refine upsampled views (bsz, 3, N, H, W) <-- Here!
    # Dense row views are synthesized.
    dense_row_views = net(dense_row_views) # refinement

    # SSL learning
    syn_sparse_indices = sample_indices(syn_combs, lf_dense_res)
    syn_views = syn_sparse_indices[-1] - syn_sparse_indices[0] + 1
    sampled_sparse_views = dense_row_views[:, :, syn_sparse_indices, :, :]
    interp_views = F.interpolate(sampled_sparse_views, (syn_views, h, w))
    target_views = dense_row_views[:, :, syn_sparse_indices[0]:syn_sparse_indices[-1] + 1, :, :]

if __name__ == "__main__":
    
    h_net = EPIVolumeNet()
    v_net = EPIVolumeNet()
    if torch.cuda.is_available():
        h_net = h_net.cuda()
        v_net = v_net.cuda()

    bsz = 16
    lf_res = 3        # N'* N' original input light field
    lf_dense_res = 7  # N'* N' -> N*N light field. !!!Variable value during training!!!
    syn_combs = find_syn_combinations(lf_res, lf_dense_res)
    h, w = 128, 128
    # Input light field data format
    lf = torch.randn(bsz, lf_res, lf_res, 3, h, w) # (bsz, N', N', 3, H, W) #

    if torch.cuda.is_available():
        lf = lf.cuda()

    # SSL training
    ssl_train_views(lf, syn_combs, h_net)  # horizontal
    lf = torch.transpose(lf, 1, 2) # transpose angular dimensions -> also transpose spatial?
    ssl_train_views(lf, syn_combs, v_net)  # vertical
    

    # Metric between interpolated views and target views
    

"""
scale = 4
lf = np.load("../data/lf.npy")
u, v, h, w, _ = lf.shape
print(lf.shape)
assert u == v # 
low_ares_lf = angular_downsample_lf(lf, scale)
print(low_ares_lf.shape)
low_ares_lf_h = get_horizontal_strip(low_ares_lf, 0)
lf_h_target = get_horizontal_strip(lf, 0)

epis = []
for i in range(h):
    epi = get_horizontal_epi_slice(low_ares_lf_h, i)
    epi_interp = cv2.resize(epi, dsize=(h, u))
    epis.append(np.expand_dims(epi_interp, 1))
epis = np.concatenate(epis, axis=1)

print(metrics.psnr(epis, lf_h_target))
#high_ares_horizontal_epi_target = get_horizontal_epi_slice(get_horizontal_strip(lf, 1), 256)
"""


"""
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(high_ares_horizontal_epi)
axarr[1].imshow(high_ares_horizontal_epi_target)
plt.show()
"""
