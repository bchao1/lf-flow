import os
import imageio
import numpy as np
from argparse import ArgumentParser
from utils import normalize, view_loss_to_dist
from utils import compute_alpha_blending, denorm_tanh
from image_utils import save_image, center_crop_resize
from torchvision.utils import save_image as tv_save_image
import torch.nn.functional as F
import math

import torch
import metrics
from train_best import get_dataset_and_loader, synthesize_lf_and_stereo, get_model, merge_lf
from utils import AverageMeter, refocus
from PIL import Image
import time

def get_err_map(lf, target_lf):
    # lf (num_views, 3, H, W)
    assert lf.shape == target_lf.shape
    err = (lf - target_lf)**2
    err = np.transpose(err[0], (0, 2, 3, 1))
    N, h, w, _ = err.shape
    lf_res = int(np.sqrt(N))

    err = np.mean(err, axis=(-1, -2, -3))

    #err = (err - np.min(err, axis=(-1, -2), keepdims=True))/(np.max(err, axis=(-1, -2), keepdims=True) - np.min(err, axis=(-1, -2), keepdims=True))
    #err = err.reshape(lf_res, lf_res, h, w)
    #err = np.transpose(err, (0, 2, 1, 3))
    #err = err.reshape(lf_res*h, lf_res*w)
    #err = (err * 255).astype(np.uint8)
    return err

def single_stereo(disparity_net, refine_net, left_path, right_path, args, i):
    import torchvision.transforms as tv_transforms

    import transforms

    crop_size = min(Image.open(left_path).size)

    t = tv_transforms.Compose([
        tv_transforms.CenterCrop(crop_size),
        tv_transforms.Resize(args.imsize),
        tv_transforms.ToTensor(),
    ])

    start = time.time()
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)
    left = t(left_img) # 0, 1
    right = t(right_img)
    left = (left - 0.5) * 2
    right = (right - 0.5) * 2
    left = left.unsqueeze(0)
    right = right.unsqueeze(0)
    #left, right = right, left

    row_idx = torch.tensor(args.lf_res // 2).float()
    # wide stereo view
    left_idx = torch.tensor(0).float()
    right_idx = torch.tensor(args.lf_res - 1).float()
    if torch.cuda.is_available():
        left = left.cuda()
        right = right.cuda()
        left_idx = left_idx.cuda()
        right_idx = right_idx.cuda()
        row_idx = row_idx.cuda()
    stereo_ratio = right_idx - left_idx

    coarse_lf_left, unit_disp1, left_attn1 = synthesize_lf_and_stereo(
            left, right, row_idx, left_idx, stereo_ratio, args.lf_res, 
            disparity_net, False, args
        )
    coarse_lf_right, unit_disp2, right_attn2 = synthesize_lf_and_stereo(
        right, left, row_idx, right_idx, stereo_ratio, args.lf_res,
        disparity_net, True, args
    )
    merged_lf = merge_lf(row_idx, left_idx, right_idx, 
        coarse_lf_left, coarse_lf_right, None, args.lf_res, args.merge_method)
        
        
    syn_lf = refine_net(merged_lf)
    syn_lf = denorm_tanh(syn_lf)
    syn_lf = syn_lf.detach().cpu().numpy()
    
    syn_lf = [Image.fromarray((np.transpose(view, (1, 2, 0)) * 255).astype(np.uint8)) for view in syn_lf[0]]
    syn_lf[0].save(os.path.join(args.output_dir, f"lf_{i}.gif"), save_all=True, append_images=syn_lf[1:], duration=100, loop=0)

    save_disparity(unit_disp1, "./temp/disp1.png")
    save_disparity(unit_disp2, "./temp/disp2.png")
    

def test_flow(disparity_net, dataset, args):
    left_idx = torch.tensor(1.0)
    right_idx = torch.tensor(5.0)
    stereo_ratio = right_idx - left_idx
    for i in range(len(dataset)):
        print(i)
        _, lf, _, _, _ = dataset.__getitem__(i)
        lf = lf.view(dataset.lf_res, dataset.lf_res, *lf.shape[1:])
        for j in range(dataset.lf_res):
            # j is row index
            left = lf[j][int(left_idx.item())]
            right = lf[j][int(right_idx.item())]
            stereo_pair = torch.cat([left, right], dim=-1).unsqueeze(0).permute(0, 3, 1, 2).float()
            row_idx = torch.tensor(j).float()
            if torch.cuda.is_available():
                stereo_pair = stereo_pair.cuda()
                row_idx = row_idx.cuda()
                left_idx = left_idx.cuda()
                right_idx = right_idx.cuda()
            left = stereo_pair[:, :3, :, :]
            right = stereo_pair[:, 3:, :, :]
            stereo_ratio = right_idx - left_idx # positive shear value
            
            coarse_lf_left, unit_disp1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                disparity_net, False, args
            )
            unit_disp1 = unit_disp1.squeeze().detach().cpu().numpy()
            unit_disp1 = (unit_disp1 - np.min(unit_disp1.ravel())) / (np.max(unit_disp1.ravel()) - np.min(unit_disp1.ravel()))
            unit_disp1 = (unit_disp1 * 255).astype(np.uint8)
            Image.fromarray(unit_disp1).save("./temp/disp_{}_{}.png".format(i, j))

def get_testing_data(dataloader):
    for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
        stereo_pair = stereo_pair.permute(0, 3, 1, 2).float().squeeze().numpy()
        stereo_pair = np.transpose(stereo_pair, (1, 2, 0))
        left = stereo_pair[:, :, :3]
        right = stereo_pair[:, :, 3:]
        left = ((left + 1) * 0.5 * 255).astype(np.uint8)
        right = ((right + 1) * 0.5 * 255).astype(np.uint8)
        Image.fromarray(left).save("./temp/left_{}.png".format(i))
        Image.fromarray(right).save("./temp/right_{}.png".format(i))

def test_horizontal_views(dataloader, args):
    print(len(dataloader.dataset))
    args.lf_res = dataloader.dataset.lf_res
    
    root = f"./temp/large_baseline_results/{args.dataset}"
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
        n = stereo_pair.shape[0]
        target_lf = target_lf[0].reshape(args.lf_res, args.lf_res, *target_lf.shape[2:])
        target_middle_row = target_lf[args.lf_res // 2]
        target_middle_row = denorm_tanh(target_middle_row) # rescale to [0, 1]
        target_middle_row = target_middle_row.cpu().numpy()
        
        imgs = (target_middle_row * 255).astype(np.uint8)
        imgs = [Image.fromarray(img) for img in imgs]
        imgs[0].save(f"./temp/lf_{i}.gif", save_all=True, append_images=imgs[1:], loop=0, duration=100)

        # read in zhang views
        views = []
        for j in range(1, args.lf_res+1):
            filename = "{}_{}.jpeg".format(i, j+3)
            img = np.array(Image.open(os.path.join(root, filename)))
            img = img * 1.0 / 255 # rescale to [0, 1]
            views.append(img)
        views = np.stack(views)

        views = np.transpose(views, (0, 3, 1, 2))
        target_middle_row = np.transpose(target_middle_row, (0, 3, 1, 2))
        psnr = metrics.psnr(views, target_middle_row)
        ssim = metrics.ssim(views, target_middle_row)
        if psnr == psnr and ssim == ssim: # check for nan
            psnr_avg.update(psnr, n)
            ssim_avg.update(ssim, n)
        
            print("PSNR: ", psnr, " | SSIM: ", ssim)
    print("Average PSNR: ", psnr_avg.avg)
    print("Average SSIM: ", ssim_avg.avg)
        
def save_disparity(disp, path):
    # single disparity map, first dimension is batch, single channel
    disp = disp[0]
    disp = (disp - disp.min()) / (disp.max() - disp.min()) # normalize to 0, 1
    tv_save_image(disp, path)
    
def run_inference(disparity_net, refine_net, dataloader, dataset, args):
    total_time = 0
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    psnr_horizontal_avg = AverageMeter()
    ssim_horizontal_avg = AverageMeter()

    for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
        start_time = time.time()

        n = stereo_pair.shape[0] # batch size
        stereo_pair = stereo_pair.permute(0, 3, 1, 2).float()
        target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
        left_idx = left_idx.float()
        right_idx = right_idx.float()
        row_idx = row_idx.float()
        if torch.cuda.is_available() and args.gpu_id >= 0:
            print("moving to cuda")
            stereo_pair = stereo_pair.cuda()
            target_lf = target_lf.cuda()
            left_idx = left_idx.cuda()
            right_idx = right_idx.cuda()
            row_idx = row_idx.cuda()
        
        left = stereo_pair[:, :3, :, :]
        right = stereo_pair[:, 3:, :, :]
        if args.stereo_ratio < 0:
            stereo_ratio = right_idx - left_idx # positive shear value
        else:
            # resampling
            stereo_ratio = torch.ones(n).to(stereo_pair.device) * args.stereo_ratio
            left_idx = torch.ones(n).to(stereo_pair.device) * (dataset.lf_res // 2 - args.stereo_ratio // 2)
            right_idx = torch.ones(n).to(stereo_pair.device) * (dataset.lf_res // 2 + args.stereo_ratio // 2)
        
        total_time -= time.time()
        coarse_lf_left, unit_disp1, left_attn1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                disparity_net, False, args
            )
        coarse_lf_right, unit_disp2, right_attn2 = synthesize_lf_and_stereo(
            right, left, row_idx, right_idx, stereo_ratio, dataset.lf_res,
            disparity_net, True, args
        )
        
        left_attn2 = 1 - right_attn2 
        left_attn = (left_attn1 + left_attn2) * 0.5 # (b, num_views, h, w)
        left_attn = left_attn.unsqueeze(2)
        #left_attn = None

        merged_lf = merge_lf(row_idx, left_idx, right_idx, 
            coarse_lf_left, coarse_lf_right, left_attn, dataset.lf_res, args.merge_method)
        
        if args.concat: # concat disparities
            stack_disp1 = torch.cat([unit_disp1] * 3, dim=1).unsqueeze(1)
            stack_disp2 = torch.cat([unit_disp2] * 3, dim=1).unsqueeze(1)
            merged_lf = torch.cat([merged_lf, stack_disp1, stack_disp2], dim=1)

        syn_lf = refine_net(merged_lf)
        syn_lf = torch.clamp(syn_lf, -1, 1)
        total_time += time.time()


        syn_lf = denorm_tanh(syn_lf)
        target_lf = denorm_tanh(target_lf)
        
        syn_lf = syn_lf.detach().cpu().numpy()
        target_lf = target_lf.detach().cpu().numpy()
        merged_lf = merged_lf.detach().cpu().numpy()
        coarse_lf_right = coarse_lf_right.detach().cpu().numpy()
        coarse_lf_left = coarse_lf_left.detach().cpu().numpy()

        middle_idx = args.lf_res * (args.lf_res // 2 )
        syn_middle_row = syn_lf[:, middle_idx:middle_idx + args.lf_res, :, :, :]
        target_middle_row = target_lf[:, middle_idx:middle_idx + args.lf_res, :, :, :]

        #err_left = get_err_map(coarse_lf_left, target_lf)
        #err_right = get_err_map(coarse_lf_right, target_lf)
        #err_merged = get_err_map(merged_lf, target_lf)
        #np.save("./temp/err_left_{}.npy".format(i), err_left)
        #np.save("./temp/err_right_{}.npy".format(i), err_right)
        #np.save("./temp/err_merged_{}.npy".format(i), err_merged)
        #print(syn_lf.shape, target_lf.shape)
        psnr = metrics.psnr(syn_lf, target_lf)
        ssim = metrics.ssim(syn_lf, target_lf)
        psnr_h = metrics.psnr(syn_middle_row, target_middle_row)
        ssim_h = metrics.ssim(syn_middle_row, target_middle_row)
        
        syn_lf = [Image.fromarray((np.transpose(view, (1, 2, 0)) * 255).astype(np.uint8)) for view in syn_lf[0]]
        syn_lf[0].save(os.path.join(args.output_dir, f"lf_{i}.gif"), save_all=True, append_images=syn_lf[1:], duration=100, loop=0)
        syn_lf[0].save(os.path.join(args.output_dir, f"lf_{i}_top_left.png"))
        syn_lf[-1].save(os.path.join(args.output_dir, f"lf_{i}_bottom_right.png"))
        save_disparity(unit_disp1, os.path.join(args.output_dir, f"lf_{i}_disp1.png"))
        save_disparity(unit_disp2, os.path.join(args.output_dir, f"lf_{i}_disp2.png"))
        tv_save_image(denorm_tanh(left), os.path.join(args.output_dir, f"lf_{i}_left.png"), padding=0)
        tv_save_image(denorm_tanh(right), os.path.join(args.output_dir, f"lf_{i}_right.png"), padding=0)
        #exit()
        #syn_lf = (syn_lf + 1) * 0.5
        #np.save('./temp/syn_{}.npy'.format(i), syn_lf)
        #syn_lf = np.transpose(syn_lf[0], (0, 2, 3, 1))
        #syn_lf = np.reshape(syn_lf, (dataset.lf_res, dataset.lf_res, *syn_lf.shape[1:]))
        #refocused_back = refocus(syn_lf, 0.5)
        #refocused_front = refocus(syn_lf, -0.5)
        #Image.fromarray(refocused_back).save(os.path.join(args.save_dir, "refocus_back_{}.png".format(i)))
        #Image.fromarray(refocused_front).save(os.path.join(args.save_dir, "refocus_front_{}.png".format(i)))

        #target_lf = (target_lf + 1) * 0.5
        #coarse_lf_left = (coarse_lf_left + 1) * 0.5
        #coarse_lf_right = (coarse_lf_right + 1) * 0.5
        #merged_lf = (merged_lf + 1) * 0.5

        # lf left right check
        #np.save('./temp/target_{}.npy'.format(i), target_lf)
        #np.save('./temp/left_{}.npy'.format(i), coarse_lf_left)
        #np.save('./temp/right_{}.npy'.format(i), coarse_lf_right)
        #np.save('./temp/merged_{}.npy'.format(i), merged_lf)

        print("PSNR: ", psnr, " | SSIM: ", ssim)
        print("PSNR (horizontal): ", psnr_h, " | SSIM (horizontal): ", ssim_h)

        psnr_avg.update(psnr, n)
        ssim_avg.update(ssim, n)
        psnr_horizontal_avg.update(psnr_h, n)
        ssim_horizontal_avg.update(ssim_h, n)
        #print("Time, ", time.time() - start_time)

    avg_time = total_time / len(dataset)
    print("Average PSNR: ", psnr_avg.avg)
    print("Average SSIM: ", ssim_avg.avg)
    print("Average PSNR (horizontal): ", psnr_horizontal_avg.avg)
    print("Average SSIM (horizontal): ", ssim_horizontal_avg.avg)
    print("Average time: ", avg_time)
    with open(f"{args.output_dir}/results.txt", "w") as file:
        print("Average PSNR: ", psnr_avg.avg, file=file)
        print("Average SSIM: ", ssim_avg.avg, file=file)
        print("Average PSNR (horizontal): ", psnr_horizontal_avg.avg, file=file)
        print("Average SSIM (horizontal): ", ssim_horizontal_avg.avg, file=file)
        print("Average time: ", avg_time, file=file)
        
def test():

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['hci', 'inria_lytro', 'inria_dlfd'])
    parser.add_argument("--imsize", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--use_epoch", type=int, default=2000)
    parser.add_argument("--max_disparity", type=float, default=10)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--merge_method", default="avg", choices=["avg", "left", "right", "alpha", "learned_alpha"])
    parser.add_argument("--disp_model", choices=["flownetc", "original"])
    parser.add_argument("--refine_model", choices=["3dcnn", "shuffle", "concat"], default="3dcnn")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--refine_hidden", type=int, default=128)
    parser.add_argument("--scale_baseline", type=float, default=1.0)
    parser.add_argument("--stereo_ratio", type=int, default=-1)
    parser.add_argument("--test_mode", type=str, choices=["normal", "stereo", "flow", "data", "horizontal"], default="normal")
    parser.add_argument("--mode", type=str, choices=["stereo_wide", "stereo_narrow"])

    args = parser.parse_args()
    args.use_crop = False
    args.use_jitter = False
    args_dict = vars(args)
    
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    if args.test_mode == "normal" and None in list(args_dict.values()):
        not_specified = [key for key in args_dict if args_dict[key] is None]
        raise ValueError("Please specify: {}".format(", ".join(not_specified)))
    if args.stereo_ratio > 0:
        assert(args.stereo_ratio % 2 == 0)
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    args.output_dir = os.path.join("./tcsvt_results/mine", args.dataset, args.name, str(args.use_epoch)) # save results to here
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare datasets
    dataset, dataloader = get_dataset_and_loader(args, train=False)

    if args.test_mode == "normal":
        args.num_views = dataset.lf_res**2
        args.lf_res = dataset.lf_res
        disparity_net, refine_net = get_model(args)


        refine_net_path = os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(args.use_epoch))
        disparity_net_path = os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(args.use_epoch))

        refine_net.load_state_dict(torch.load(refine_net_path, map_location=f"cuda:{args.gpu_id}"))
        disparity_net.load_state_dict(torch.load(disparity_net_path, map_location=f"cuda:{args.gpu_id}"))

        if torch.cuda.is_available() and args.gpu_id >= 0:
            torch.cuda.set_device(args.gpu_id)

        if torch.cuda.is_available() and args.gpu_id >= 0:
            print("Running using GPU")
            refine_net = refine_net.cuda()
            disparity_net = disparity_net.cuda()
        
        refine_net.eval()
        disparity_net.eval()

    if args.test_mode == "normal":
        run_inference(disparity_net, refine_net, dataloader, dataset, args)
    elif args.test_mode == "flow":
        test_flow(disparity_net, dataset, args)
    elif args.test_mode == "stereo": # HERE. Read in left-right stereo images
        i = 2
        left_path = "./temp/left_rect.png"#f"production/testing_data/hci_testing_data/left_{i}.png"
        right_path = "./temp/right_rect.png"#f"production/testing_data/hci_testing_data/right_{i}.png"
        single_stereo(disparity_net, refine_net, left_path, right_path, args, i)
        
    elif args.test_mode == "data":
        get_testing_data(dataloader)
    elif args.test_mode == "horizontal":
        test_horizontal_views(dataloader, args)


if __name__ == '__main__':
    with torch.no_grad():
        test()