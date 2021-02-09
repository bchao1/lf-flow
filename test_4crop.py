import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from argparse import ArgumentParser
import metrics 
from torchvision.utils import save_image as tv_save_image

from models.net_siggraph import Network
from train_4crop import get_dataset_and_loader, get_input_features, warp_to_view
from utils import AverageMeter, denorm_tanh

def main():
    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--max_disparity", type=float, default=21) # specified in paper -21 ~ 21
    parser.add_argument("--disparity_levels", type=int, default=100) # specified in paper
    parser.add_argument("--scale_disparity", type=float, default=4)
    
    parser.add_argument("--gpu_id", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'], default='hci')
    parser.add_argument("--fold", type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--use_epoch", type=int, default=1000)

    args = parser.parse_args()
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    os.makedirs(os.path.join(args.save_dir, args.dataset, args.name), exist_ok=True)
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    args.output_dir = os.path.join("temp", args.name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.use_crop = False

    dataset, dataloader = get_dataset_and_loader(args, train=False)
    args.lf_res = dataset.lf_res
    args.num_views = args.lf_res**2

    refine_net = Network(in_channels=3*4+1, out_channels=3)
    depth_net = Network(in_channels=args.disparity_levels*2, out_channels=1)

    refine_net_path = os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(args.use_epoch))
    depth_net_path = os.path.join(args.save_dir, "ckpt", "depth_{}.ckpt".format(args.use_epoch))

    refine_net.load_state_dict(torch.load(refine_net_path, map_location="cpu"))
    depth_net.load_state_dict(torch.load(depth_net_path, map_location="cpu"))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
#
    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        depth_net = depth_net.cuda()
    
    refine_net.eval()
    depth_net.eval()

    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    disparities = torch.linspace(-args.max_disparity, args.max_disparity, args.disparity_levels)
    total_time = 0
    for i, (corner_views, target_lf) in enumerate(dataloader):
        n, _, h, w, _ = corner_views.shape
        
        corner_views = corner_views.permute(0, 1, 4, 2, 3).float()
        target_lf = target_lf.permute(0, 1, 2, 5, 3, 4).float()
        
        if torch.cuda.is_available():
            corner_views = corner_views.cuda()
            target_lf = target_lf.cuda()
        
        psnr_views = 0
        ssim_views = 0
        for u in range(dataset.lf_res):
            for v in range(dataset.lf_res):
                # warping here!
                print(u, v)
                target_pos_i = torch.tensor(u).float()
                target_pos_j = torch.tensor(v).float()
                if torch.cuda.is_available():
                    target_pos_i = target_pos_i.cuda()
                    target_pos_j = target_pos_j.cuda()
                
                target_view = target_lf[:, u, v, :, :, :]

                total_time -= time.time() # critical section

                features = get_input_features(corner_views, target_pos_i, target_pos_j, dataset.lf_res, disparities)
                depth = depth_net(features) * args.scale_disparity # (b, 1, H, W)
                coarse_view = warp_to_view(corner_views, target_pos_i, target_pos_j, depth, dataset.lf_res)

                joined = torch.cat([coarse_view, depth.unsqueeze(0).unsqueeze(0)], dim=1)   
                syn_view = refine_net(joined)
                
                if u == 0 and v == 0:
                    tv_save_image(denorm_tanh(syn_view), os.path.join(args.output_dir, "tl_syn_{}.png".format(i)))
                    tv_save_image(denorm_tanh(target_view), os.path.join(args.output_dir, "tl_target_{}.png".format(i)))
                elif u == args.lf_res - 1 and v == args.lf_res - 1:
                    tv_save_image(denorm_tanh(syn_view), os.path.join(args.output_dir, "br_syn_{}.png".format(i)))
                    tv_save_image(denorm_tanh(target_view), os.path.join(args.output_dir, "br_target_{}.png".format(i)))

                total_time += time.time()

                syn_view = syn_view.detach().cpu().numpy()
                target_view = target_view.detach().cpu().numpy().squeeze()

                psnr = metrics.psnr(syn_view, target_view)
                ssim = metrics.ssim(syn_view, target_view, mode = 1)
                
                psnr_views += psnr
                ssim_views += ssim

        psnr_avg.update(psnr_views / args.num_views, n)
        ssim_avg.update(ssim_views / args.num_views, n)
        print("PSNR: ", psnr, " | SSIM: ", ssim)

    avg_time = total_time / len(dataset)
    print("Average PSNR: ", psnr_avg.avg)
    print("Average SSIM: ", ssim_avg.avg)
    print("Inference time: ", avg_time)


        
        

if __name__ == '__main__':
    #test()
    with torch.no_grad():
        main()