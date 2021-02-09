import os
import time
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image as tv_save_image
from argparse import ArgumentParser
from utils import AverageMeter, denorm_tanh
from image_utils import lf_to_multiview
import metrics
from models.lf_net import LFRefineNet, DepthNet
from train_single_image import get_dataset_and_loader, synthsize_lf_from_single_image

def main():

    parser = ArgumentParser()
    
    # train setting
    
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--disparity_scale", type=float, default=10)
    
    parser.add_argument("--gpu_id", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'], default='hci')
    parser.add_argument("--fold", type=int, choices=list(range(5)), help="Kth-fold for Stanford Dataset")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--use_epoch", type=int, default=1000)

    args = parser.parse_args()
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    args.use_crop = False
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    args.output_dir = os.path.join("temp", args.name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # set up
    dataset, dataloader = get_dataset_and_loader(args, train=False)
    args.views = dataset.lf_res**2

    refine_net = LFRefineNet(in_channels=dataset.lf_res**2 * 2, out_channels=dataset.lf_res**2)
    depth_net = DepthNet(views=dataset.lf_res**2)

    refine_net_path = os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(args.use_epoch))
    depth_net_path = os.path.join(args.save_dir, "ckpt", "depth_{}.ckpt".format(args.use_epoch))

    refine_net.load_state_dict(torch.load(refine_net_path, map_location="cpu"))
    depth_net.load_state_dict(torch.load(depth_net_path, map_location="cpu"))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        depth_net = depth_net.cuda()
    
    refine_net.eval()
    depth_net.eval()

    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    syn_results = []
    total_time = 0
    for i, (center_image, target_lf) in enumerate(dataloader):

        n, h, w, _ = center_image.shape
        center_image = center_image.permute(0, 3, 1, 2).float()
        target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
        
        if torch.cuda.is_available():
            center_image = center_image.cuda()
            target_lf = target_lf.cuda()
        
        total_time -= time.time()
        depth = depth_net(center_image) # (b, N, H, W)
        
        # warp by depth
        coarse_lf = synthsize_lf_from_single_image(center_image, depth, dataset.lf_res, args)
        depth_cat = torch.cat([depth.unsqueeze(2)] * 3, dim=2) #(b, N, 3, H, W)
        joined = torch.cat([coarse_lf, depth_cat], dim=1)            
        syn_lf = refine_net(joined)
        total_time += time.time()

        top_left_syn = syn_lf[0][0, :]
        tv_save_image(denorm_tanh(top_left_syn), os.path.join(args.output_dir, "tl_syn_{}.png".format(i)))
        top_left_target = target_lf[0][0, :]
        tv_save_image(denorm_tanh(top_left_target), os.path.join(args.output_dir, "tl_target_{}.png".format(i)))
        bottom_right_syn = syn_lf[0][-1, :]
        tv_save_image(denorm_tanh(bottom_right_syn), os.path.join(args.output_dir, "br_syn_{}.png".format(i)))
        bottom_right_target = target_lf[0][-1, :]
        tv_save_image(denorm_tanh(bottom_right_target), os.path.join(args.output_dir, "br_target_{}.png".format(i)))

        syn_lf = syn_lf.detach().cpu().numpy()
        depth = depth.squeeze().detach().cpu().numpy()
        
        #dmin = np.min(depth, axis=(1, 2), keepdims=True)
        #dmax = np.max(depth, axis=(1, 2), keepdims=True)
        #dnorm = (depth - dmin) / (dmax - dmin)
        #dnorm = dnorm.reshape(dataset.lf_res, dataset.lf_res, *dnorm.shape[1:], 1)
        #mv = lf_to_multiview(dnorm).squeeze()
        #mv = (mv * 255).astype(np.uint8)
        #Image.fromarray(mv).save("./temp/depths_{}.png".format(i))
        

        syn_results.append(syn_lf)
        target_lf = target_lf.detach().cpu().numpy()
        
        psnr = metrics.psnr(syn_lf, target_lf)
        ssim = metrics.ssim(syn_lf, target_lf)
        psnr_avg.update(psnr, n)
        ssim_avg.update(ssim, n)
        print("PSNR: ", psnr, " | SSIM: ", ssim)
    avg_time = total_time / len(dataset)
    print("Average PSNR: ", psnr_avg.avg)
    print("Average SSIM: ", ssim_avg.avg)
    print("Inference time: ", avg_time)

if __name__ == '__main__':
    #test()
    with torch.no_grad():
        main()