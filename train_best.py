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
from utils import Dummy
from image_utils import lf_to_multiview, save_image
from image_utils import sobel_filter_batch, sample_stereo_index
from metrics import ColorConstancyLoss, WeightedReconstructionLoss, TVLoss

import models.flownet2.models as flownet_models
# import here! lfnet of ver2
from models.lf_net import LFRefineNet, LFRefineNetV2, DisparityNet
import transforms

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
            fold = args.fold,
            use_crop = args.use_crop
        )
    else:
        raise ValueError("dataset [{}] not supported".format(args.dataset))
    print("Dataset size: {}".format(dataset.__len__()))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, drop_last=False)
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

def predict_rot_flow(view1, view2, stereo_ratio, dispartiy_net, rev, max_disp):
    batch = torch.cat([view1, view2], dim=1)
    n = batch.shape[0]
    flow, attn = dispartiy_net(batch)
    flow = flow * max_disp # scale [-1, -1] to [-max_disp, max_disp]
    if rev:
        flow = flow * -1
    unit_flow = flow / stereo_ratio.view(n, 1, 1, 1) # scale the flow to unit step
    return unit_flow


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

# Flow consistency: PROBLEM
def warp_flow(unit_disp, row_idx, left_idx, right_idx, dispartiy_net, lf, lf_res, max_disp, rev=False):
    b, _, c, h, w = lf.shape
    lf = lf.view(b, lf_res, lf_res, c, h, w)
    new_row_idx = np.random.randint(lf_res, size=b)

    batch_idx = np.arange(b)
    left_idx = left_idx.cpu().numpy()
    right_idx = right_idx.cpu().numpy()
    stereo_ratio = right_idx - left_idx
    stereo_ratio = torch.tensor(stereo_ratio).to(lf.device)

    left = lf[batch_idx, new_row_idx, left_idx, :, :, :]
    right = lf[batch_idx, new_row_idx, right_idx, :, :, :]

    # test left to right or right to left
    if rev:
        stereo_pairs = torch.cat([left, right], dim=1)
    else:
        stereo_pairs = torch.cat([right, left], dim=1)

    new_disp = dispartiy_net(stereo_pairs) * max_disp

    # negation if right to left
    if rev:
        new_disp *= -1
    new_unit_disp = new_disp / stereo_ratio.view(b, 1, 1, 1)

    row_shift = torch.tensor(new_row_idx).to(row_idx.device) - row_idx
    row_shift = row_shift.view(b, 1, 1)
    dx = unit_disp.squeeze() * 0
    dy = unit_disp.squeeze() * row_shift
    warped_unit_disp = warp_image_batch(unit_disp, dx, dy)
    return warped_unit_disp, new_unit_disp

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
    parser.add_argument("--edge_loss", type=str, choices=['l1', 'l2'], default='l1')
    parser.add_argument("--edge_loss_w", type=float, default=0.1)
    parser.add_argument('--consistency_w', type=float, default=1)
    parser.add_argument('--flow_consistency_w', type=float, default=0.05)
    parser.add_argument("--tv_loss_w", type=float, default=0.01)
    parser.add_argument("--rot_loss_w", type=float)
    
    parser.add_argument("--gpu_id", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'], default='hci')
    parser.add_argument("--fold", default=-1, type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
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
    
    if args.edge_loss == 'l1':
        edge_criterion = WeightedReconstructionLoss(loss_func = nn.L1Loss())
    elif args.edge_loss == 'l2':
        edge_criterion = WeightedReconstructionLoss(loss_func = nn.MSELoss())
    else:
        raise ValueError("Edge preserving loss {} not supported!".format(args.edge_loss))
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
            """
            for j, lf in enumerate(target_lf):
                lf = lf.numpy()
                lf = lf.reshape(dataset.lf_res, dataset.lf_res, *lf.shape[1:])
                print(lf.shape)
                lf = (lf + 1) * 0.5
                mv = lf_to_multiview(lf) # [0, 1]
                mv = (255 * mv).astype(np.uint8)
                Image.fromarray(mv).save("./temp/lf{}.png".format(j))
            exit()
            """
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
            left_rot90 = torch.rot90(left, 1, [2, 3])
            right_rot90 = torch.rot90(right, 1, [2, 3])

            stereo_ratio = right_idx - left_idx # positive shear value

            coarse_lf_left, unit_disp1, left_attn1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                dispartiy_net, False, args
            )
            coarse_lf_right, unit_disp2, right_attn2 = synthesize_lf_and_stereo(
                right, left, row_idx, right_idx, stereo_ratio, dataset.lf_res,
                dispartiy_net, True, args
            )

            #unit_disp1_rot90_predict = predict_rot_flow(
            #    left_rot90, right_rot90, stereo_ratio, dispartiy_net, False, args.max_disparity
            #)
            #unit_disp2_rot90_predict = predict_rot_flow(
            #    right_rot90, left_rot90, stereo_ratio, dispartiy_net, True, args.max_disparity
            #)
            #unit_disp1_rot90_target = torch.rot90(unit_disp1, 1, [2, 3])
            #unit_disp2_rot90_target = torch.rot90(unit_disp2, 1, [2, 3])

            
            
            left_attn2 = 1 - right_attn2 
            left_attn = (left_attn1 + left_attn2) * 0.5 # (b, num_views, h, w)
            left_attn = left_attn.unsqueeze(2)

            merged_lf = merge_lf(row_idx, left_idx, right_idx, 
                coarse_lf_left, coarse_lf_right, left_attn, dataset.lf_res, args.merge_method)

            # flow-consistency loss
            #warped_flow1, new_flow1 = warp_flow(unit_disp1, 
            #    row_idx, left_idx, right_idx, 
            #    dispartiy_net, target_lf, dataset.lf_res, args.max_disparity, False)
            #warped_flow2, new_flow2 = warp_flow(unit_disp2, 
            #    row_idx, left_idx, right_idx, 
            #    dispartiy_net, target_lf, dataset.lf_res, args.max_disparity, True)

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
            flow_consistency_loss = torch.tensor(0.0)#(criterion(warped_flow1, new_flow1, 1) + criterion(warped_flow2, new_flow2, 1)) * args.flow_consistency_w
            consistency_loss = (criterion(coarse_lf_left, target_lf, 1) + criterion(coarse_lf_right, target_lf, 1)) * 0.5 * args.consistency_w
            tv_loss = tv_criterion(unit_disp1) + tv_criterion(unit_disp2)
            rot_loss = torch.tensor(0)#(criterion(unit_disp1_rot90_predict, unit_disp1_rot90_target, 1) + criterion(unit_disp2_rot90_predict, unit_disp2_rot90_target)) * 0.5 * args.rot_loss_w

            loss = lf_loss + consistency_loss + flow_consistency_loss + tv_loss + rot_loss
        

            optimizer_refine.zero_grad()
            optimizer_disp.zero_grad()
            loss.backward()
            optimizer_refine.step()
            optimizer_disp.step()

            lf_loss_log.append(lf_loss.item())
            print("Epoch {:5d}, iter {:2d} | consistency: {:10f} | rot {:10f} | lf {:10f} | tv {:10f}".format(
                e, i, consistency_loss.item(), rot_loss.item(), lf_loss.item(), tv_loss.item()
            ))

        plot_loss_logs(lf_loss_log, "lf_loss", os.path.join(args.save_dir, 'plots'))
        if e % args.save_epochs == 0:
            torch.save(refine_net.state_dict(), os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(e)))
            torch.save(dispartiy_net.state_dict(), os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(e)))
            

if __name__ == '__main__':
    #test()
    main()