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

from torchvision.utils import save_image as tv_save_image
import torchvision.transforms as tv_transforms

from argparse import ArgumentParser

from lf_datasets import HCIDataset, INRIADataset, StanfordDataset
from utils import warp_image_batch, generate_lf_batch_single_image
from utils import plot_loss_logs
from utils import denorm_tanh, normalize
from utils import compute_alpha_blending, get_weight_map
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch
from metrics import TVLoss, DepthConsistencyLoss

from models.flownet import FlowNetS, FlowNetC
from models.lf_net import LFRefineNet, DepthNet
import transforms


def get_dataset_and_loader(args, train):
    transform = tv_transforms.Compose([
        transforms.ToTensor(),
        transforms.NormalizeRange(-1, 1) # for tanh residual addition
    ])

    if args.dataset == 'hci':
        dataset = HCIDataset(
            root = "../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = "4crop"
        )
    elif args.dataset == 'inria':
        dataset = INRIADataset(
            root = "../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5", 
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = "4crop"
        )
    elif args.dataset == 'stanford':
        if args.fold is None:
            raise ValueError("Please specify fold for Stanford Dataset!")
        dataset = StanfordDataset(
            root = "../../../mnt/data2/bchao/lf/stanford/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            fold = args.fold,
            mode = "4crop"
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return dataset, dataloader
    
def synthsize_lf_from_crops(center_image, depths, lf_res, args):
    depths = depths * args.disparity_scale # scale 
    lf = generate_lf_batch_single_image(center_image, depths, lf_res)
    return lf

def main():
    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--disparity_levels", type=float, default=100) # specified in paper
    parser.add_argument("--use_crop", action="store_true")

    # Losses and regularizations

    parser.add_argument("--recon_loss", type=str, choices=['l1', 'l2'], default='l1')
    # regularization in paper
    parser.add_argument("--tv_loss_w", type=float)
    parser.add_argument("--c_loss_w", type=float)
    
    parser.add_argument("--gpu_id", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'], default='hci')
    parser.add_argument("--fold", type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
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
    copyfile("./bash_run_scripts/train_single_image.sh", os.path.join(args.save_dir, "train_single_image.sh"))

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=True)
    
    # criterion = reconstruction loss
    if args.recon_loss == 'l1':
        criterion = nn.L1Loss()
    elif args.recon_loss == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Reconstruction loss {} not supported!".format(args.recon_loss))
    
    # total variation regularization
    tv_criterion = TVLoss(w = args.tv_loss_w)
    consistency_criterion = DepthConsistencyLoss(w = args.c_loss_w)

    refine_net = LFRefineNet(views=dataset.lf_res**2, with_depth=True)
    depth_net = DepthNet(views=dataset.lf_res**2)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        depth_net = depth_net.cuda()

        # Loss functions
        criterion = criterion.cuda()
        tv_criterion = tv_criterion.cuda()
        consistency_criterion = consistency_criterion.cuda()

    optimizer_refine = torch.optim.Adam(refine_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    optimizer_depth = torch.optim.Adam(depth_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)


    lf_loss_log = []
    for e in range(1, args.train_epochs + 1):
        start = time.time()
        for i, (center_image, target_lf) in enumerate(dataloader):
            n, h, w, _ = center_image.shape

            center_image = center_image.permute(0, 3, 1, 2).float()
            target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
            
            if torch.cuda.is_available():
                center_image = center_image.cuda()
                target_lf = target_lf.cuda()

            depth = depth_net(center_image) # (b, N, H, W)
            # warp by depth
            coarse_lf = synthsize_lf_from_single_image(center_image, depth, dataset.lf_res, args)
            depth_cat = torch.cat([depth.unsqueeze(2)] * 3, dim=2) #(b, N, 3, H, W)

            joined = torch.cat([coarse_lf, depth_cat], dim=1)            
            syn_lf = refine_net(joined)

            lf_loss = criterion(syn_lf, target_lf)
            tv_loss = tv_criterion(depth)
            c_loss = consistency_criterion(depth)

            loss = lf_loss + tv_loss + c_loss

            optimizer_refine.zero_grad()
            optimizer_depth.zero_grad()
            loss.backward()
            optimizer_refine.step()
            optimizer_depth.step()

            lf_loss_log.append(lf_loss.item())

            print("Epoch {:5d}, iter {:2d} | lf {:10f} | tv {:10f} | c {:10f} |".format(
                e, i, lf_loss.item(), tv_loss.item(), c_loss.item()
            ))

        plot_loss_logs(lf_loss_log, "lf_loss", os.path.join(args.save_dir, 'plots'))
        if e % args.save_epochs == 0:
            # checkpointing
            #np.save(os.path.join(args.save_dir, "results", "syn_lf_{}.npy".format(e)), syn_lf[0].detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "results", "target_lf_{}.npy".format(e)), target_lf[0].detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp1_{}.npy".format(e)), disp1.detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp2_{}.npy".format(e)), disp2.detach().cpu().numpy())
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(depth_net.state_dict(), os.path.join(args.save_dir, "ckpt", "depth_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()