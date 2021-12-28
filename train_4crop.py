import os
import time
import json
from shutil import copyfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.utils import save_image as tv_save_image
import torchvision.transforms as tv_transforms

from argparse import ArgumentParser

from lf_datasets import HCIDataset, INRIADataset, INRIA_DLFD_Dataset
from utils import warp_image_batch, generate_lf_batch_single_image
from utils import plot_loss_logs, AverageMeter
from utils import denorm_tanh, normalize
from utils import compute_alpha_blending, get_weight_map
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch
from metrics import TVLoss, DepthConsistencyLoss

from models.net_siggraph import Network
import transforms


def get_dataset_and_loader(args, train):
    transform = tv_transforms.Compose([
        transforms.ToTensor(),
        transforms.NormalizeRange(-1, 1) # for tanh residual addition
    ])

    if args.dataset == 'hci':
        dataset = HCIDataset(
            root = "/mount/data/hci/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = args.mode
        )
    elif args.dataset == 'inria_lytro':
        dataset = INRIADataset(
            root = "/mount/data/inria_lytro/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = args.mode
        )
    elif args.dataset == "inria_dlfd":
        dataset = INRIA_DLFD_Dataset(
            root = "/mount/data/inria_dlfd/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = args.mode
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, drop_last=False, pin_memory=True, num_workers=16)
    return dataset, dataloader
    
def get_input_features(corner_views, target_i, target_j, lf_res, disparity_levels, mode):
    # corner views (b, 4, 3, h, w)
    # target view (b, 3, h, w)
    # target i b,
    # target j b, 
    # disparities L
    b, _, _, h, w = corner_views.shape
    if mode == "4crop":
        corner_indices = [(0, 0), (0, lf_res - 1), (lf_res - 1, 0), (lf_res - 1, lf_res - 1)]
    elif mode == "2crop_wide":
        corner_indices = [(lf_res // 2, 0), (lf_res // 2, lf_res - 1)]
    elif mode == "2crop_narrow":
        corner_indices = [(lf_res // 2, lf_res // 2 - 1), (lf_res // 2, lf_res // 2 + 1)]

    means = []
    stds = []
    for d in disparity_levels:
        warped = []
        for idx, (i, j) in enumerate(corner_indices):
            shift_i = target_i - i
            shift_j = target_j - j
            disp = torch.ones(b, h, w).to(corner_views.device) * d
            dx = disp * shift_j.view(b, 1, 1)
            dy = disp * shift_i.view(b, 1, 1)
            corner_view = corner_views[:, idx, :, :, :]
            syn = warp_image_batch(corner_view, dx, dy)
            warped.append(syn.unsqueeze(1))
        warped = torch.cat(warped, dim=1).view(b, len(corner_indices) * 3, h, w)
        warped_mean = torch.mean(warped, dim=1)
        warped_std = torch.std(warped, dim=1)
        means.append(warped_mean.unsqueeze(1))
        stds.append(warped_std.unsqueeze(1))
    means = torch.cat(means, dim=1)
    stds = torch.cat(stds, dim=1)
    feat = torch.cat([means, stds], dim=1)
    return feat

def warp_to_view(corner_views, target_i, target_j, disp, lf_res, mode):
    b, _, _, h, w = corner_views.shape
    if mode == "4crop":
        corner_indices = [(0, 0), (0, lf_res - 1), (lf_res - 1, 0), (lf_res - 1, lf_res - 1)]
    elif mode == "2crop_wide":
        corner_indices = [(lf_res // 2, 0), (lf_res // 2, lf_res - 1)]
    elif mode == "2crop_narrow":
        corner_indices = [(lf_res // 2, lf_res // 2 - 1), (lf_res // 2, lf_res // 2 + 1)]
        
    warped = []
    for idx, (i, j) in enumerate(corner_indices):
        shift_i = target_i - i
        shift_j = target_j - j
        dx = disp * shift_j.view(b, 1, 1)
        dy = disp * shift_i.view(b, 1, 1)
        corner_view = corner_views[:, idx, :, :, :]
        syn = warp_image_batch(corner_view, dx, dy)
        warped.append(syn)
    warped = torch.cat(warped, dim=1)
    return warped

def get_exp_name(args):
    name = f"{args.dataset}_{args.name}_kalantari-{args.mode}"
    return name

def main():

    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--max_disparity", type=float, default=21) # specified in paper -21 ~ 21
    parser.add_argument("--disparity_levels", type=int, default=100) # specified in paper
    parser.add_argument("--scale_disparity", type=float, default=4)
    parser.add_argument("--use_crop", action="store_true")
    parser.add_argument("--mode", type=str, choices=["2crop_wide", "2crop_narrow", "4crop"])

    # Losses and regularizations

    parser.add_argument("--recon_loss", type=str, choices=['l1', 'l2'], default='l1')
    # regularization in paper
    parser.add_argument("--tv_loss_w", type=float)
    parser.add_argument("--c_loss_w", type=float)
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'inria_dlfd', 'inria_lytro'], default='hci')
    parser.add_argument("--fold", type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    exp_name = get_exp_name(args)
    writer = SummaryWriter(comment=exp_name)
    
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.name), exist_ok=True)
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    os.makedirs(os.path.join(args.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    # write config to file
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as file:
        json.dump(vars(args), file)
    copyfile("./bash_run_scripts/train_4crop.sh", os.path.join(args.save_dir, "train_4crop.sh"))

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=True)
    
    # criterion = reconstruction loss
    if args.recon_loss == 'l1':
        criterion = nn.L1Loss()
    elif args.recon_loss == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Reconstruction loss {} not supported!".format(args.recon_loss))

    if args.mode == "4crop":
        refine_net = Network(in_channels=3*4+3, out_channels=3) # 3N+3
    elif "2crop" in args.mode:
        refine_net = Network(in_channels=3*2+3, out_channels=3) # 3N+3
    depth_net = Network(in_channels=args.disparity_levels*2, out_channels=1) # 2*L

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        depth_net = depth_net.cuda()

        # Loss functions
        criterion = criterion.cuda()

    optimizer_refine = torch.optim.Adam(refine_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    optimizer_depth = torch.optim.Adam(depth_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    disparities = torch.linspace(-args.max_disparity, args.max_disparity, args.disparity_levels)
    loss_log = []
    
    for e in range(1, args.train_epochs + 1):
        start = time.time()
        epoch_loss = AverageMeter()
        for i, (corner_views, target_view, target_pos_i, target_pos_j) in enumerate(dataloader):
            n, _, h, w, _ = corner_views.shape
            
            corner_views = corner_views.permute(0, 1, 4, 2, 3).float()
            target_view = target_view.permute(0, 3, 1, 2).float()
            
            if torch.cuda.is_available():
                corner_views = corner_views.cuda()
                target_view = target_view.cuda()
                target_pos_i = target_pos_i.cuda()
                target_pos_j = target_pos_j.cuda()

            # warping here!
            features = get_input_features(corner_views, target_pos_i, target_pos_j, dataset.lf_res, disparities, args.mode)

            depth = depth_net(features) * args.scale_disparity # (b, 1, H, W)
            coarse_view = warp_to_view(corner_views, target_pos_i, target_pos_j, depth, dataset.lf_res, args.mode)

            target_pos_i_feat = target_pos_i.reshape(target_pos_i.shape[0], 1, 1, 1).repeat(1, 1, h, w)
            target_pos_j_feat = target_pos_j.reshape(target_pos_j.shape[0], 1, 1, 1).repeat(1, 1, h, w)
            joined = torch.cat([coarse_view, depth.unsqueeze(1), target_pos_i_feat, target_pos_j_feat], dim=1) 
            syn_view = refine_net(joined)

            loss = criterion(syn_view, target_view)
            #err = ((syn_view - target_view)**2).mean(1)

            optimizer_refine.zero_grad()
            optimizer_depth.zero_grad()
            loss.backward()
            optimizer_refine.step()
            optimizer_depth.step()
            loss_log.append(loss.item())

            print("Epoch {:5d}, iter {:2d} | loss {:10f} |".format(
                e, i, loss.item()
            ))

            epoch_loss.update(loss.item())

        #plot_loss_logs(loss_log, "loss", os.path.join(args.save_dir, 'plots'))
        writer.add_scalar("loss", epoch_loss.avg, e)
        writer.add_images("syn_view", denorm_tanh(syn_view), e)
        writer.add_images("target_view", denorm_tanh(target_view), e)
        

        if e % args.save_epochs == 0:
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(depth_net.state_dict(), os.path.join(args.save_dir, "ckpt", "depth_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()