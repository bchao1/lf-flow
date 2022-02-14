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
from tqdm import tqdm

from lf_datasets import HCIDataset, INRIADataset, INRIA_DLFD_Dataset
from utils import warp_image_batch, generate_lf_batch
from utils import plot_loss_logs
from utils import denorm_tanh, normalize
from utils import compute_alpha_blending, get_weight_map
from utils import Dummy, AverageMeter
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch, sample_stereo_index
from metrics import ColorConstancyLoss, WeightedReconstructionLoss, TVLoss

# import here! lfnet of ver2
from models.lf_net import LFRefineNet, LFRefineNetV2, DisparityNet
import transforms
from torch.utils.tensorboard import SummaryWriter

def get_model(args):
    if args.refine_model == "concat":
        refine_net = LFRefineNet(in_channels=args.num_views+2, out_channels=args.num_views, hidden=args.refine_hidden) # also pad disparity
    elif args.refine_model == "3dcnn":
        refine_net = LFRefineNet(in_channels=args.num_views, out_channels=args.num_views, hidden=args.refine_hidden) # also pad disparity
    elif args.refine_model == "shuffle":
        if args.imsize % args.lf_res != 0:
            raise ValueError("When using shuffle, imsize({}) should be divisible by({})!".format(args.imsize, args.lf_res))
        refine_net = LFRefineNetV2(args.num_views, args.num_views, args.num_views)
    
    if args.disp_model == "flownetc":
        args.fp16 = False
        dispartiy_net = flownet_models.FlowNetC.FlowNetC(args)
    elif args.disp_model == "original":
        dispartiy_net = DisparityNet(args.num_views)
    
    return dispartiy_net, refine_net

def get_dataset_and_loader(args, train):
    if args.use_jitter:
        transform = tv_transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
            transforms.RandomSaturation(),
            transforms.NormalizeRange(-1, 1) # for tanh residual addition
        ])
    else:
        transform = tv_transforms.Compose([
            transforms.ToTensor(),
            transforms.NormalizeRange(-1, 1) # for tanh residual addition
        ])

    if args.dataset == 'hci':
        dataset = HCIDataset(
            #root = "/mount/data/hci/dataset.h5",
            root = "/mnt/data2/bchao/lf/tcsvt_datasets/hci/dataset.h5",
            train = train,
            im_size = args.imsize,
            transform = transform,
            use_all = False,
            use_crop = args.use_crop,
            mode = args.mode
        )
    elif args.dataset == 'inria_dlfd':
        dataset = INRIA_DLFD_Dataset(
            root = "/mnt/data2/bchao/lf/tcsvt_datasets/inria_dlfd/dataset.h5",
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
            #root = "../tcsvt_datasets/inria_real/dataset.h5",
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

def merge_lf(row_idx, left_idx, right_idx, left, right, left_attn, lf_res, method):
    if method == "alpha":
        merged_lf = compute_alpha_blending(row_idx, left_idx, right_idx, left, right, lf_res)
    elif method == "avg":
        # Naive blend light field
        merged_lf = (left + right) * 0.5
    elif method == "left":
        merged_lf = left
    elif method == "right":
        merged_lf = right
    elif method == "learned_alpha":
        merged_lf = left_attn * left + (1 - left_attn) * right
    else:
        print("Merge method {} not supported".format(method))
    return merged_lf

def synthesize_lf_and_stereo(view1, view2, row_idx, view1_idx, stereo_ratio, lf_res, dispartiy_net, rev, args):
    batch = torch.cat([view1, view2], dim=1)
    n = batch.shape[0]

    if args.disp_model == "flownetc":
        flow = dispartiy_net(batch)[0]
        flow = F.interpolate(flow, scale_factor=4)
        flow = torch.mean(flow, dim=1).unsqueeze(1)
    elif args.disp_model == "original":
        flow, attn = dispartiy_net(batch)
        flow = flow * args.max_disparity # scale [-1, -1] to [-max_disp, max_disp]

    # test here!
    if rev:
        flow = flow * -1
    
    unit_flow = flow / stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step
    #empty_flow = torch.zeros_like(flow)
    coarse_lf = generate_lf_batch(view1, row_idx, view1_idx, unit_flow.squeeze(), lf_res, args.scale_baseline)
    #syn_view2 = warp_image_batch(view1, flow_horizontal * stereo_ratio.view(n, 1, 1), empty_flow)
    return coarse_lf, unit_flow, attn#, syn_view2
    
def get_syn_stereo_flow(lf, lf_res, dispartiy_net, rev, args):
    # lf (b, num_vies, c, h, w)
    b, num_views, c, h, w = lf.shape
    assert num_views == lf_res**2
    lf = lf.view(b, lf_res, lf_res, c, h, w)
    row_idx, idx1, idx2 = sample_stereo_index(lf_res)
    if rev:
        idx1, idx2 = idx2, idx1
    view1 = lf[:, row_idx, idx1, :, :, :]
    view2 = lf[:, row_idx, idx2, :, :, :]
    batch = torch.cat([view1, view2], dim=1)

    if args.disp_model == "original":
        flow = dispartiy_net(batch) * args.max_disparity
    elif args.disp_model == "flownetc":
        flow = dispartiy_net(batch)[0]# * args.max_disparity # dont need to scale
        flow = F.interpolate(flow, scale_factor=4) # upsample flow
        flow = torch.mean(flow, dim=1).unsqueeze(1)
    
    unit_flow = flow / (idx2 - idx1)
    return unit_flow

def get_exp_name(args):
    name = f"{args.dataset}_{args.name}_ref-{args.refine_model}_merge-{args.merge_method}_tv-{args.tv_loss_w}"
    return name

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
    parser.add_argument("--use_jitter", action="store_true")

    parser.add_argument("--disp_model", choices=["flownetc", "original"], default="original")
    parser.add_argument("--refine_model", choices=["3dcnn", "shuffle", "concat"], default="3dcnn")
    parser.add_argument("--refine_hidden", type=int, default=128)

    # Losses and regularizations
    parser.add_argument("--merge_method", default="avg", choices=["avg", "left", "right", "alpha", "learned_alpha"])
    parser.add_argument("--use_weighted_view", action="store_true")

    parser.add_argument("--recon_loss", type=str, choices=['l1', 'l2'], default='l1')
    parser.add_argument('--consistency_w', type=float, default=1)
    parser.add_argument("--tv_loss_w", type=float, default=0.01)
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'inria_lytro', 'inria_dlfd'], default='hci')
    parser.add_argument("--fold", default=-1, type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--mode", type=str, choices=["stereo_wide", "stereo_narrow"])

    args = parser.parse_args()
    exp_name = get_exp_name(args)

    writer = SummaryWriter(comment=exp_name)


    args.stereo_ratio = -1
    args.scale_baseline = 1
    args_dict = vars(args)
    if None in list(args_dict.values()):
        not_specified = [key for key in args_dict if args_dict[key] is None]
        raise ValueError("Please specify: {}".format(", ".join(not_specified)))
    print(args)
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.name), exist_ok=True)
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    os.makedirs(os.path.join(args.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    # write config to file
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as file:
        json.dump(args_dict, file)
    #copyfile("./bash_run_scripts/train.sh", os.path.join(args.save_dir, "train.sh"))

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=True)
    
    # criterion = reconstruction loss
    if args.recon_loss == 'l1':
        criterion = WeightedReconstructionLoss(loss_func = nn.L1Loss())
    elif args.recon_loss == 'l2':
        criterion = WeightedReconstructionLoss(loss_func = nn.MSELoss())
    else:
        raise ValueError("Reconstruction loss {} not supported!".format(args.recon_loss))
    
    tv_criterion = TVLoss(w = args.tv_loss_w)

    args.num_views = dataset.lf_res**2
    args.lf_res = dataset.lf_res

    dispartiy_net, refine_net = get_model(args)

    #print("disparity net: ")
    #print(dispartiy_net)
    #print("refine net: ")
    #print(refine_net)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    
    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        dispartiy_net = dispartiy_net.cuda()

        # Loss functions
        criterion = criterion.cuda()
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
    tv_loss_avg = AverageMeter()
    lf_loss_avg = AverageMeter()
    consistency_loss_avg = AverageMeter()
    #stereo, target_lf, left_idx, right_idx = next(iter(dataloader))
    for e in range(1, args.train_epochs + 1):
        start = time.time()
        for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(tqdm(dataloader)):
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

            stereo_ratio = right_idx - left_idx # positive shear value

            coarse_lf_left, unit_disp1, left_attn1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                dispartiy_net, False, args
            )
            coarse_lf_right, unit_disp2, right_attn2 = synthesize_lf_and_stereo(
                right, left, row_idx, right_idx, stereo_ratio, dataset.lf_res,
                dispartiy_net, True, args
            )

            
            left_attn2 = 1 - right_attn2 
            left_attn = (left_attn1 + left_attn2) * 0.5 # (b, num_views, h, w)
            left_attn = left_attn.unsqueeze(2)

            merged_lf = merge_lf(row_idx, left_idx, right_idx, 
                coarse_lf_left, coarse_lf_right, left_attn, dataset.lf_res, args.merge_method)

            if args.refine_model == "concat":
                stack_disp1 = torch.cat([unit_disp1] * 3, dim=1).unsqueeze(1)
                stack_disp2 = torch.cat([unit_disp2] * 3, dim=1).unsqueeze(1)
                joined = torch.cat([merged_lf, stack_disp1, stack_disp2], dim=1)
                syn_lf = refine_net(joined)
            else:
                syn_lf = refine_net(merged_lf)

            weight_map = 1
            if args.use_weighted_view:
                weight_map = get_weight_map(row_idx, left_idx, right_idx, dataset.lf_res)
            
            lf_loss = criterion(syn_lf, target_lf, weight_map)
            consistency_loss = (criterion(coarse_lf_left, target_lf, 1) + criterion(coarse_lf_right, target_lf, 1)) * 0.5 * args.consistency_w
            tv_loss = tv_criterion(unit_disp1) + tv_criterion(unit_disp2)

            loss = lf_loss + consistency_loss + tv_loss
        

            optimizer_refine.zero_grad()
            optimizer_disp.zero_grad()
            loss.backward()
            optimizer_refine.step()
            optimizer_disp.step()

            lf_loss_avg.update(lf_loss.item())
            consistency_loss_avg.update(consistency_loss.item())
            tv_loss_avg.update(tv_loss.item())

        print("Epoch {:5d}, iter {:2d} | consistency: {:10f} | lf {:10f} | tv {:10f}".format(
            e, i, consistency_loss_avg.avg, lf_loss_avg.avg, tv_loss_avg.avg
        ))

        #plot_loss_logs(lf_loss_log, "lf_loss", os.path.join(args.save_dir, 'plots'))
        writer.add_scalar("losses/lf", lf_loss_avg.avg, e)
        writer.add_scalar("losses/tv", tv_loss_avg.avg, e)
        writer.add_scalar("losses/consistency", consistency_loss_avg.avg, e)

        if e % 100 == 0:
            disp1 = denorm_tanh(unit_disp1) #[0,1]
            disp2 = denorm_tanh(unit_disp2) #[0,1]
            left_lf_view = denorm_tanh(coarse_lf_left[:, 0])
            right_lf_view = denorm_tanh(coarse_lf_right[:, 0])
            target_lf_view = denorm_tanh(target_lf[:, 0])

            writer.add_images("disp/l2r", disp1, e)
            writer.add_images("disp/r2l", disp2, e)

            writer.add_images("views/left", left_lf_view, e)
            writer.add_images("views/right", right_lf_view, e)
            writer.add_images("views/target", target_lf_view, e)

        if e % args.save_epochs == 0:
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(dispartiy_net.state_dict(), os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()