mport os
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

def single_stereo(disparity_net, refine_net, left_path, right_path, args):
    import torchvision.transforms as tv_transforms

    import transforms

    crop_size = min(Image.open(left_path).size)

    t = tv_transforms.Compose([
        tv_transforms.CenterCrop(crop_size),
        tv_transforms.Resize(args.imsize),
        tv_transforms.ToTensor(),
    ])

    i = "bunny"

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
    syn_lf = torch.clamp(syn_lf, -1, 1)
    unit_disp1 = unit_disp1.squeeze().detach().cpu().numpy()
    unit_disp2 = unit_disp2.squeeze().detach().cpu().numpy()
    unit_disp1 = (unit_disp1 - np.min(unit_disp1.ravel())) / (np.max(unit_disp1.ravel()) - np.min(unit_disp1.ravel()))
    unit_disp2 = (unit_disp2 - np.min(unit_disp2.ravel())) / (np.max(unit_disp2.ravel()) - np.min(unit_disp2.ravel()))
    unit_disp1 = (unit_disp1 * 255).astype(np.uint8)
    unit_disp2 = (unit_disp2 * 255).astype(np.uint8)
    Image.fromarray(unit_disp1).save("./temp/disp_{}_1.png".format(i))
    Image.fromarray(unit_disp2).save("./temp/disp_{}_2.png".format(i))
    
    syn_lf = syn_lf.detach().cpu().numpy()
    merged_lf = merged_lf.detach().cpu().numpy()
    coarse_lf_right = coarse_lf_right.detach().cpu().numpy()
    coarse_lf_left = coarse_lf_left.detach().cpu().numpy()

    syn_lf = (syn_lf + 1) * 0.5
    syn_lf = np.transpose(syn_lf[0], (0, 2, 3, 1))
    syn_lf = np.reshape(syn_lf, (args.lf_res, args.lf_res, *syn_lf.shape[1:]))
    refocused_back = refocus(syn_lf, 0.5)
    refocused_front = refocus(syn_lf, -0.5)
    Image.fromarray(refocused_back).save("./temp/refocus_back_{}.png".format(i))
    Image.fromarray(refocused_front).save("./temp/refocus_front_{}.png".format(i))
    np.save("./temp/syn.npy", syn_lf)


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
    print(args.dataset)
    root = "./data/zhang/"
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
        n = stereo_pair.shape[0]
        middle_idx = args.lf_res * (args.lf_res // 2)
        target_middle_row = target_lf[:, middle_idx + 1:middle_idx + args.lf_res, :, :, :]
        target_middle_row = target_middle_row.cpu().numpy()[0]
        target_middle_row = (target_middle_row + 1) * 00.5
        # read in zhang views
        views = []
        for j in range(1, args.lf_res):
            filename = "{}_{}_{}.jpeg".format(args.dataset, i, j)
            img = np.array(Image.open(os.path.join(root, filename)))
            img = img * 1.0 / 255
            views.append(img)
        views = np.stack(views)
        #views = dataloader.dataset.transform(views).numpy()

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
        
    
def run_inference(disparity_net, refine_net, dataloader, dataset, args):
    total_time = 0
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    psnr_horizontal_avg = AverageMeter()
    ssim_horizontal_avg = AverageMeter()

    for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
        n = stereo_pair.shape[0] # batch size
        stereo_pair = stereo_pair.permute(0, 3, 1, 2).float()
        target_lf = target_lf.permute(0, 1, 4, 2, 3).float()
        left_idx = left_idx.float()
        right_idx = right_idx.float()
        row_idx = row_idx.float()
        if torch.cuda.is_available() and args.gpu_id >= 0:
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


        unit_disp1 = unit_disp1.squeeze().detach().cpu().numpy()
        unit_disp2 = unit_disp2.squeeze().detach().cpu().numpy()
        unit_disp1 = (unit_disp1 - np.min(unit_disp1.ravel())) / (np.max(unit_disp1.ravel()) - np.min(unit_disp1.ravel()))
        unit_disp2 = (unit_disp2 - np.min(unit_disp2.ravel())) / (np.max(unit_disp2.ravel()) - np.min(unit_disp2.ravel()))
        unit_disp1 = (unit_disp1 * 255).astype(np.uint8)
        unit_disp2 = (unit_disp2 * 255).astype(np.uint8)
        Image.fromarray(unit_disp1).save(os.path.join(args.output_dir, "disp_{}_1.png".format(i)))
        Image.fromarray(unit_disp2).save(os.path.join(args.output_dir, "disp_{}_2.png".format(i)))
        left_attn = left_attn.squeeze().unsqueeze(1)
        #tv_save_image(left_attn, "./temp/attn_{}.png".format(i))

        # save top-left view
        top_left_syn = syn_lf[0][0, :]
        tv_save_image(denorm_tanh(top_left_syn), os.path.join(args.output_dir, "tl_syn_{}.png".format(i)))
        top_left_target = target_lf[0][0, :]
        tv_save_image(denorm_tanh(top_left_target), os.path.join(args.output_dir, "tl_target_{}.png".format(i)))
        bottom_right_syn = syn_lf[0][-1, :]
        tv_save_image(denorm_tanh(bottom_right_syn), os.path.join(args.output_dir, "br_syn_{}.png".format(i)))
        bottom_right_target = target_lf[0][-1, :]
        tv_save_image(denorm_tanh(bottom_right_target), os.path.join(args.output_dir, "br_target_{}.png".format(i)))


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

        syn_lf = (syn_lf + 1) * 0.5
        #np.save('./temp/syn_{}.npy'.format(i), syn_lf)
        syn_lf = np.transpose(syn_lf[0], (0, 2, 3, 1))
        syn_lf = np.reshape(syn_lf, (dataset.lf_res, dataset.lf_res, *syn_lf.shape[1:]))
        refocused_back = refocus(syn_lf, 0.5)
        refocused_front = refocus(syn_lf, -0.5)
        Image.fromarray(refocused_back).save(os.path.join(args.save_dir, "refocus_back_{}.png".format(i)))
        Image.fromarray(refocused_front).save(os.path.join(args.save_dir, "refocus_front_{}.png".format(i)))

        target_lf = (target_lf + 1) * 0.5
        coarse_lf_left = (coarse_lf_left + 1) * 0.5
        coarse_lf_right = (coarse_lf_right + 1) * 0.5
        merged_lf = (merged_lf + 1) * 0.5

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

    avg_time = total_time / len(dataset)
    print("Average PSNR: ", psnr_avg.avg)
    print("Average SSIM: ", ssim_avg.avg)
    print("Average PSNR (horizontal): ", psnr_horizontal_avg.avg)
    print("Average SSIM (horizontal): ", ssim_horizontal_avg.avg)
    print("Average time: ", avg_time)

def test():

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'])
    parser.add_argument("--imsize", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--use_epoch", type=int, default=1000)
    parser.add_argument("--max_disparity", type=float, default=10)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--merge_method", default="avg", choices=["avg", "left", "right", "alpha", "learned_alpha"])
    parser.add_argument("--disp_model", choices=["flownetc", "original"])
    parser.add_argument("--refine_model", choices=["3dcnn", "shuffle", "concat"], default="3dcnn")
    parser.add_argument("--gpu_id", type=int, choices=[-1, 0, 1])
    parser.add_argument("--refine_hidden", type=int, default=128)
    parser.add_argument("--scale_baseline", type=float, default=1.0)
    parser.add_argument("--stereo_ratio", type=int, default=-1)
    parser.add_argument("--mode", type=str, choices=["normal", "stereo", "flow", "data", "horizontal"], default="normal")

    args = parser.parse_args()
    args.use_crop = False
    args.use_jitter = False
    args_dict = vars(args)
    
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    if None in list(args_dict.values()):
        not_specified = [key for key in args_dict if args_dict[key] is None]
        raise ValueError("Please specify: {}".format(", ".join(not_specified)))
    if args.stereo_ratio > 0:
        assert(args.stereo_ratio % 2 == 0)
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    args.output_dir = os.path.join("temp", args.name) # save results to here
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Prepare datasets
    dataset, dataloader = get_dataset_and_loader(args, train=False)

    args.num_views = dataset.lf_res**2
    args.lf_res = dataset.lf_res
    disparity_net, refine_net = get_model(args)


    refine_net_path = os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(args.use_epoch))
    disparity_net_path = os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(args.use_epoch))

    refine_net.load_state_dict(torch.load(refine_net_path))
    disparity_net.load_state_dict(torch.load(disparity_net_path))

    if torch.cuda.is_available() and args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)

    if torch.cuda.is_available() and args.gpu_id >= 0:
        refine_net = refine_net.cuda()
        disparity_net = disparity_net.cuda()
    
    refine_net.eval()
    disparity_net.eval()

    if args.mode == "normal":
        run_inference(disparity_net, refine_net, dataloader, dataset, args)
    elif args.mode == "flow":
        test_flow(disparity_net, dataset, args)
    elif args.mode == "stereo": # HERE. Read in left-right stereo images
        left_path = "./data/left.jpeg"
        right_path = "./data/right.jpeg"

        single_stereo(disparity_net, refine_net, left_path, right_path, args)
    elif args.mode == "data":
        get_testing_data(dataloader)
    elif args.mode == "horizontal":
        test_horizontal_views(dataloader, args)


if __name__ == '__main__':
    with torch.no_grad():
        test()

