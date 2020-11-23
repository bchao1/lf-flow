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

from lf_datasets import HCIDataset
from utils import warp_image_batch, generate_lf_batch
from utils import plot_loss_logs
from utils import denorm_tanh, normalize
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch
from metrics import ColorConstancyLoss

from models.flownet import FlowNetS, FlowNetC
from models.lf_net import LFRefineNet, DisparityNet
import transforms

def test():

    im_size = 128
    batch_size = 8
    dataset = HCIDataset(
        root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5",
        train=True,
        im_size=im_size,
        use_all=False
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    dispartiy_net = FlowNetS.flownets_bn().cuda()
    refine_net = LFRefineNet(in_channels=dataset.lf_res*dataset.lf_res).cuda()



    stereo_pair, target_lf, left_idx, right_idx = next(iter(dataloader))
    n = stereo_pair.shape[0]

    stereo_pair = stereo_pair.permute(0, 3, 1, 2).cuda().float()
    target_lf = target_lf.permute(0, 1, 4, 2, 3).cuda().float()
    left_idx = left_idx.cuda().float()
    right_idx = right_idx.cuda().float()

    left = stereo_pair[:, :3, :, :]
    right = stereo_pair[:, 3:, :, :]
    
    stereo_ratio = right_idx - left_idx

    flow = dispartiy_net(stereo_pair)[0] # largest resolution flow
    flow = F.interpolate(flow, size=left.size()[-2:], mode='bilinear', align_corners=False)
    flow /= stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step
    flow_horizontal = flow[:, 0, :, :] # we only need to use the horizontal flow for warping
    flow_vertical = flow[:, 1, :, :]

    #warped = warp_image_torch(left, flow_horizontal * left_s.view(n, 1, 1), flow_vertical * row_s.view(n, 1, 1))
    #print(warped.shape, v.shape)
    coarse_lf_from_left = generate_lf_batch(left, left_idx, flow_horizontal, dataset.lf_res)
    syn_lf = refine_net(syn_lf)

    empty_flow = torch.zeros_like(flow_horizontal)
    syn_right = warp_image_batch(left, flow_horizontal * stereo_ratio, empty_flow)
    
    loss = criterion(lf, target_lf)
    loss.backward()

def get_dataset_and_loader(args, train):
    transform = tv_transforms.Compose([
        transforms.ToTensor(),
        transforms.NormalizeRange(-1, 1) # for tanh residual addition
    ])

    if args.dataset == 'hci':
        dataset = HCIDataset(
            root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5",
            train=train,
            im_size=args.imsize,
            transform = transform,
            use_all=False,
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return dataset, dataloader

def synthesize_lf_and_stereo(view1, view2, view1_idx, stereo_ratio, lf_res, dispartiy_net, rev, args):
    batch = torch.cat([view1, view2], dim=1)
    n = batch.shape[0]

    flow = dispartiy_net(batch)
    if rev:
        flow = -flow

    flow = flow * args.max_disparity # scale [-1, -1] to [-max_disp, max_disp]
    unit_flow = flow / stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step

    #empty_flow = torch.zeros_like(flow)
    coarse_lf = generate_lf_batch(view1, view1_idx, unit_flow.squeeze(), lf_res)
    #syn_view2 = warp_image_batch(view1, flow_horizontal * stereo_ratio.view(n, 1, 1), empty_flow)
    return coarse_lf, flow#, syn_view2
    

def main():
    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=int, default=100)
    parser.add_argument("--max_disparity", type=float, default=10)

    parser.add_argument("--recon_loss", type=str, choices=['l1', 'l2'], default='l1')
    parser.add_argument("--edge_loss_w", type=float, default=0.1)
    parser.add_argument("--color_loss_w", type=float, default=0.1)
    
    parser.add_argument("--gpu_id", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'lytro'], default='hci')
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
    args.save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(os.path.join(args.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    # write config to file
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as file:
        json.dump(vars(args), file)
    copyfile("train.sh", os.path.join(args.save_dir, "train.sh"))

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=True)
    
    # criterion = reconstruction loss
    if args.recon_loss == 'l1':
        criterion = nn.L1Loss()
    elif args.recon_loss == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Loss {} not supported!".format(args.recon_loss))
    l1loss = nn.L1Loss()
    color_criterion = ColorConstancyLoss(patch_size=args.imsize // 4)
    norm01 = transforms.Normalize01()
    
    refine_net = LFRefineNet(in_channels=dataset.lf_res**2)
    dispartiy_net = DisparityNet()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        dispartiy_net = dispartiy_net.cuda()
        criterion = criterion.cuda()
        l1loss = l1loss.cuda()
    
    optimizer = torch.optim.Adam(
        [
            {'params': refine_net.parameters()},
            {'params': dispartiy_net.parameters()}
        ],
        lr=args.lr, betas=[0.9, 0.999]
    )


    lf_loss_log = []
    edge_loss_log = []
    #stereo, target_lf, left_idx, right_idx = next(iter(dataloader))
    for e in range(1, args.train_epochs + 1):
        start = time.time()
        for i, (stereo_pair, target_lf, left_idx, right_idx) in enumerate(dataloader):
            stereo_pair = stereo_pair.permute(0, 3, 1, 2).float()
            target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
            left_idx = left_idx.float()
            right_idx = right_idx.float()

            if torch.cuda.is_available():
                stereo_pair = stereo_pair.cuda()
                target_lf = target_lf.cuda()
                left_idx = left_idx.cuda()
                right_idx = right_idx.cuda()
            
            left = stereo_pair[:, :3, :, :]
            right = stereo_pair[:, 3:, :, :]

            stereo_ratio = right_idx - left_idx # positive shear value

            coarse_lf_left, disp1 = synthesize_lf_and_stereo(
                left, right, left_idx, stereo_ratio, dataset.lf_res, 
                dispartiy_net, False, args
            )
            #print(disp1.shape)
            #exit()
            coarse_lf_right, disp2 = synthesize_lf_and_stereo(
                right, left, right_idx, stereo_ratio, dataset.lf_res,
                dispartiy_net, True, args
            )

            merged_lf = (coarse_lf_left + coarse_lf_right) * 0.5
            syn_lf = refine_net(merged_lf)
            edge_syn_lf = sobel_filter_batch(syn_lf.view(-1, *syn_lf.shape[2:]), mode=1)
            edge_target_lf = sobel_filter_batch(target_lf.view(-1, *syn_lf.shape[2:]), mode=1)
            #edge_l_x, edge_l_y = sobel_filter_batch(coarse_lf_left.view(-1, *coarse_lf_left.shape[2:]))
            #edge_r_x, edge_r_y = sobel_filter_batch(coarse_lf_right.view(-1, *coarse_lf_right.shape[2:]))
            #edge_lf_target = sobel_filter_batch(target_lf.view(-1, *target_lf.shape[2:]))
            #a = edge_lf_left[0].detach().cpu().numpy()
            #a = normalize(a) * 255
            #save_image(a, 'test.png')
            #a = edge_lf_target[0].detach().cpu().numpy()
            #a = normalize(a) * 255
            #save_image(a, 'target.png')
            #exit()

            consistency_loss = criterion(coarse_lf_left, coarse_lf_right)
            lf_loss = criterion(syn_lf, target_lf)
            edge_loss = l1loss(edge_syn_lf, edge_target_lf)
            color_constancy_loss = color_criterion(syn_lf) # mean color of a patch should be same across views

            loss = lf_loss + consistency_loss + \
                edge_loss * args.edge_loss_w + \
                color_constancy_loss * args.color_loss_w
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lf_loss_log.append(lf_loss.item())
            edge_loss_log.append(edge_loss.item())
            print("Epoch {:5d}, iter {:2d} | consistency: {:10f} | lf {:10f} | edge {:10f}".format(
                e, i, consistency_loss.item(), lf_loss.item(), edge_loss.item()
            ))

        plot_loss_logs(lf_loss_log, "lf_loss", os.path.join(args.save_dir, 'plots'))
        plot_loss_logs(edge_loss_log, "edge_loss", os.path.join(args.save_dir, 'plots'))
        if e % args.save_epochs == 0:
            # checkpointing
            np.save(os.path.join(args.save_dir, "results", "syn_lf_{}.npy".format(e)), syn_lf[0].detach().cpu().numpy())
            np.save(os.path.join(args.save_dir, "results", "target_lf_{}.npy".format(e)), target_lf[0].detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp1_{}.npy".format(e)), disp1.detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp2_{}.npy".format(e)), disp2.detach().cpu().numpy())
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(dispartiy_net.state_dict(), os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()