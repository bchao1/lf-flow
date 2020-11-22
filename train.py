import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tv_transforms

from argparse import ArgumentParser

from lf_datasets import HCIDataset
from utils import warp_image_batch, generate_lf_batch
from utils import plot_loss_logs
from image_utils import lf_to_multiview, save_image
from models.flownet import FlowNetS, FlowNetC
from models.lf_net import LFRefineNet
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

def get_dataset_and_loader(args):
    transform = tv_transforms.Compose([
        transforms.ToTensor(),
        transforms.NormalizeRange(-1, 1) # for tanh residual addition
    ])

    if args.dataset == 'hci':
        dataset = HCIDataset(
            root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5",
            train=20,
            im_size=args.imsize,
            transform = transform,
            use_all=False,
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return dataset, dataloader

def synthesize_lf_and_stereo(view1, view2, view1_idx, stereo_ratio, lf_res, dispartiy_net, rev=False):
    batch = torch.cat([view1, view2], dim=1)
    n = batch.shape[0]

    flow = dispartiy_net(batch)[0] # largest resolution flow
    if rev:
        flow = -flow

    flow = F.interpolate(flow, size=view1.size()[-2:], mode='bilinear', align_corners=False)
    flow /= stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step
    flow_horizontal = flow[:, 0, :, :] # we only need to use the horizontal flow for warping
    #flow_vertical = flow[:, 1, :, :]
    empty_flow = torch.zeros_like(flow_horizontal)

    coarse_lf = generate_lf_batch(view1, view1_idx, flow_horizontal, lf_res)
    #syn_view2 = warp_image_batch(view1, flow_horizontal * stereo_ratio.view(n, 1, 1), empty_flow)
    return coarse_lf#, syn_view2
    

def main():
    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'lytro'], default='hci')
    parser.add_argument("--save_dir", type=str, default="experiments")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    dataset, dataloader = get_dataset_and_loader(args)
    criterion = nn.MSELoss()
    
    refine_net = LFRefineNet(in_channels=dataset.lf_res**2)
    dispartiy_net = FlowNetS.flownets_bn()

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        dispartiy_net = dispartiy_net.cuda()
        criterion = criterion.cuda()
    
    optimizer = torch.optim.Adam(
        [
            {'params': refine_net.parameters()},
            {'params': dispartiy_net.parameters()}
        ],
        lr=args.lr, betas=[0.9, 0.999]
    )


    lf_loss_log = []
    stereo_loss_log = []
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

            coarse_lf_left = synthesize_lf_and_stereo(
                left, right, left_idx, stereo_ratio, dataset.lf_res, 
                dispartiy_net, rev=False
            )
            coarse_lf_right = synthesize_lf_and_stereo(
                right, left, right_idx, stereo_ratio, dataset.lf_res,
                dispartiy_net, rev=True
            )

            merged_lf = (coarse_lf_left + coarse_lf_right) * 0.5
            syn_lf = merged_lf
            #syn_lf = refine_net(merged_lf)
            #print(syn_lf.shape)

            l2r_flow_loss = torch.tensor(0).cuda()#criterion(syn_right, right)
            r2l_flow_loss = torch.tensor(0).cuda()#criterion(syn_left, left)
            lf_loss = criterion(syn_lf, target_lf)

            loss = 5 * (l2r_flow_loss + r2l_flow_loss) + lf_loss
            loss.backward()
            optimizer.step()

            lf_loss_log.append(lf_loss.item())
            stereo_loss_log.append(l2r_flow_loss.item() + r2l_flow_loss.item())
            print("Epoch {:5d}, iter {:2d} | l2r {:10f} | r2l {:10f} | lf {:10f} |".format(
                e, i, l2r_flow_loss.item(), r2l_flow_loss.item(), lf_loss.item()
            ))

        plot_loss_logs(lf_loss_log, "lf_loss", args.save_dir)
        plot_loss_logs(stereo_loss_log, "stereo_loss", args.save_dir)
        if e % args.save_epochs == 0:
            # checkpointing
            np.save(os.path.join(args.save_dir, "syn_lf_{}.npy".format(e)), syn_lf.detach().cpu().numpy())
            np.save(os.path.join(args.save_dir, "target_lf_{}.npy".format(e)), target_lf.detach().cpu().numpy())
            

if __name__ == '__main__':
    #test()
    main()