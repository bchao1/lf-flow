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
from utils import warp_image_batch, generate_lf_batch
from utils import plot_loss_logs
from utils import denorm_tanh, normalize
from utils import compute_alpha_blending, get_weight_map
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch
from metrics import ColorConstancyLoss, WeightedReconstructionLoss, TVLoss

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
            root = "../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop
        )
    elif args.dataset == 'inria':
        dataset = INRIADataset(
            root = "../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5", 
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop
        )
    elif args.dataset == 'stanford':
        if args.fold is None:
            raise ValueError("Please specify fold for Stanford Dataset!")
        dataset = StanfordDataset(
            root = "../../../mnt/data2/bchao/lf/stanford/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            fold = args.fold
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return dataset, dataloader

def synthesize_lf_and_stereo(view1, view2, row_idx, view1_idx, stereo_ratio, lf_res, dispartiy_net, rev, args):
    batch = torch.cat([view1, view2], dim=1)
    n = batch.shape[0]

    flow = dispartiy_net(batch)
    if rev:
        flow = flow * -1
    flow = flow * args.max_disparity # scale [-1, -1] to [-max_disp, max_disp]
    unit_flow = flow / stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step
    #empty_flow = torch.zeros_like(flow)
    coarse_lf = generate_lf_batch(view1, row_idx, view1_idx, unit_flow.squeeze(), lf_res)
    #syn_view2 = warp_image_batch(view1, flow_horizontal * stereo_ratio.view(n, 1, 1), empty_flow)
    return coarse_lf, unit_flow#, syn_view2
    

def main():
    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    #parser.add_argument("--warup_to", type=int, default=100)
    parser.add_argument("--lr_decay_times", type=int, default=5) # decay learning rate??
    parser.add_argument("--max_disparity", type=float, default=10)
    parser.add_argument("--use_crop", action="store_true")

    # Losses and regularizations
    parser.add_argument("--use_alpha_blending", action="store_true")
    parser.add_argument("--use_weighted_view", action="store_true")

    parser.add_argument("--recon_loss", type=str, choices=['l1', 'l2'], default='l1')
    parser.add_argument("--edge_loss", type=str, choices=['l1', 'l2'], default='l1')
    parser.add_argument("--edge_loss_w", type=float, default=0.1)
    parser.add_argument('--consistency_w', type=float, default=1)
    parser.add_argument("--tv_loss_w", type=float, default=0.01)
    
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
    copyfile("./bash_run_scripts/train.sh", os.path.join(args.save_dir, "train.sh"))

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=True)
    
    # criterion = reconstruction loss
    if args.recon_loss == 'l1':
        criterion = WeightedReconstructionLoss(loss_func = nn.L1Loss())
    elif args.recon_loss == 'l2':
        criterion = WeightedReconstructionLoss(loss_func = nn.MSELoss())
    else:
        raise ValueError("Reconstruction loss {} not supported!".format(args.recon_loss))
    
    if args.edge_loss == 'l1':
        edge_criterion = WeightedReconstructionLoss(loss_func = nn.L1Loss())
    elif args.edge_loss == 'l2':
        edge_criterion = WeightedReconstructionLoss(loss_func = nn.MSELoss())
    else:
        raise ValueError("Edge preserving loss {} not supported!".format(args.edge_loss))
    tv_criterion = TVLoss(w = args.tv_loss_w)
    args.num_views = dataset.lf_res**2

    refine_net = LFRefineNet(in_channels=args.num_views+2, out_channels=args.num_views)
    dispartiy_net = DisparityNet()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        dispartiy_net = dispartiy_net.cuda()

        # Loss functions
        criterion = criterion.cuda()
        edge_criterion = edge_criterion.cuda()
        tv_criterion = tv_criterion.cuda()
        #color_criterion = color_criterion.cuda()
    
    #optimizer = torch.optim.Adam(
    #    [
    #        {'params': refine_net.parameters()},
    #        {'params': dispartiy_net.parameters()}
    #    ],
    #    lr=args.lr, betas=[0.9, 0.999]
    #)
    optimizer_refine = torch.optim.Adam(refine_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    optimizer_disp = torch.optim.Adam(dispartiy_net.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)


    lf_loss_log = []
    edge_loss_log = []
    #stereo, target_lf, left_idx, right_idx = next(iter(dataloader))
    for e in range(1, args.train_epochs + 1):
        start = time.time()
        for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
            n, h, w, _ = stereo_pair.shape

            stereo_pair = stereo_pair.permute(0, 3, 1, 2).float()
            target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
            left_idx = left_idx.float()
            right_idx = right_idx.float()
            row_idx = row_idx.float()

            if torch.cuda.is_available():
                stereo_pair = stereo_pair.cuda()
                target_lf = target_lf.cuda()
                left_idx = left_idx.cuda()
                right_idx = right_idx.cuda()
                row_idx = row_idx.cuda()
            
            left = stereo_pair[:, :3, :, :]
            right = stereo_pair[:, 3:, :, :]
            #left = (left + 1) * 0.5
            #right = (right + 1) * 0.5
            #tv_save_image(left, "left.png")
            #tv_save_image(right, "right.png")
            #exit()

            stereo_ratio = right_idx - left_idx # positive shear value

            coarse_lf_left, unit_disp1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                dispartiy_net, False, args
            )
            coarse_lf_right, unit_disp2 = synthesize_lf_and_stereo(
                right, left, row_idx, right_idx, stereo_ratio, dataset.lf_res,
                dispartiy_net, True, args
            )
            
            if args.use_alpha_blending:
                merged_lf = compute_alpha_blending(left_idx, right_idx, coarse_lf_left, coarse_lf_right, dataset.lf_res)
            else:
                # Naive blend light field
                merged_lf = (coarse_lf_left + coarse_lf_right) * 0.5
            stack_disp1 = torch.cat([unit_disp1] * 3, dim=1).unsqueeze(1)
            stack_disp2 = torch.cat([unit_disp2] * 3, dim=1).unsqueeze(1)
            joined = torch.cat([merged_lf, stack_disp1, stack_disp2], dim=1)
            #merged_lf = coarse_lf_left
            #lf = merged_lf[1]
            #lf = (lf + 1) * 0.5
            #print(disp1)
            #disp = disp1[0].squeeze().detach().cpu().numpy()
            #disp = normalize(disp) * 255
            #save_image(disp, 'test.png')
            #tv_save_image(disp, 'test.png')
            #exit()
            syn_lf = refine_net(joined)
            edge_syn_lf = sobel_filter_batch(syn_lf.view(-1, *syn_lf.shape[2:]), mode=1)
            edge_target_lf = sobel_filter_batch(target_lf.view(-1, *syn_lf.shape[2:]), mode=1)
            edge_syn_lf = edge_syn_lf.view(n, dataset.lf_res**2, 1, h, w)
            edge_target_lf = edge_target_lf.view(n, dataset.lf_res**2, 1, h, w)
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
            
            # Weight for each synthsized view
            weight_map = 1
            if args.use_weighted_view:
                weight_map = get_weight_map(row_idx, left_idx, right_idx, dataset.lf_res)
            
            lf_loss = criterion(syn_lf, target_lf, weight_map)
            consistency_loss = criterion(coarse_lf_left, coarse_lf_right, 1) * args.consistency_w
            edge_loss = edge_criterion(edge_syn_lf, edge_target_lf, weight_map) * args.edge_loss_w
            tv_loss = tv_criterion(unit_disp1) + tv_criterion(unit_disp2)
            loss = lf_loss + consistency_loss + edge_loss + tv_loss
            
            optimizer_refine.zero_grad()
            optimizer_disp.zero_grad()
            loss.backward()
            optimizer_refine.step()
            optimizer_disp.step()

            lf_loss_log.append(lf_loss.item())
            edge_loss_log.append(edge_loss.item())
            print("Epoch {:5d}, iter {:2d} | consistency: {:10f} | lf {:10f} | edge {:10f} | tv {:10f}".format(
                e, i, consistency_loss.item(), lf_loss.item(), edge_loss.item(), tv_loss.item()
            ))

        plot_loss_logs(lf_loss_log, "lf_loss", os.path.join(args.save_dir, 'plots'))
        plot_loss_logs(edge_loss_log, "edge_loss", os.path.join(args.save_dir, 'plots'))
        if e % args.save_epochs == 0:
            # checkpointing
            #np.save(os.path.join(args.save_dir, "results", "syn_lf_{}.npy".format(e)), syn_lf[0].detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "results", "target_lf_{}.npy".format(e)), target_lf[0].detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp1_{}.npy".format(e)), disp1.detach().cpu().numpy())
            #np.save(os.path.join(args.save_dir, "disp2_{}.npy".format(e)), disp2.detach().cpu().numpy())
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(dispartiy_net.state_dict(), os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()