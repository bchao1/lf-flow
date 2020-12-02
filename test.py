import os
import numpy as np
from argparse import ArgumentParser
from utils import normalize, view_loss_to_dist
from image_utils import save_image
import torch.nn.functional as F

def test_naive_flow():
    datasets = ['hci', 'inria', 'stanford']
    for d in datasets:
        print(d)
        v = np.load('temp/psnr_v_{}.npy'.format(d))
        h = np.load('temp/psnr_h_{}.npy'.format(d))
        print("v mean: {}".format(np.mean(v)))
        print("h mean: {}".format(np.mean(h)))
    v = np.load('temp/psnr_v_{}.npy'.format('hci'))
    h = np.load('temp/psnr_h_{}.npy'.format('hci'))
    print(np.argmax((h - v)**2))
    print(h[6])
    print(v[6])

def plot_loss_against_dist(dist, loss):
    import matplotlib 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(dist, loss)
    plt.savefig("test.png")
    plt.close()

def run_inference():
    import torch
    import metrics
    from models.lf_net import DisparityNet, LFRefineNet
    from train import get_dataset_and_loader, synthesize_lf_and_stereo
    from utils import AverageMeter

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['hci', 'stanford', 'inria'])
    parser.add_argument("--imsize", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--name", type=str)
    parser.add_argument("--use_epoch", type=int, default=1000)
    parser.add_argument("--max_disparity", type=float, default=10)
    parser.add_argument("--runs", type=int, default=1) # average result over 5 inference runs
    parser.add_argument("--fold", type=int, default=0)

    args = parser.parse_args()
    args.use_crop = False
    args_dict = vars(args)
    
    if args.dataset == 'stanford':
        args.name = args.name + "_fold{}".format(args.fold)
    if None in list(args_dict.values()):
        not_specified = [key for key in args_dict if args_dict[key] is None]
        raise ValueError("Please specify: {}".format(", ".join(not_specified)))
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.name)


    # Prepare datasets
    dataset, dataloader = get_dataset_and_loader(args, train=False)

    refine_net = LFRefineNet(views=dataset.lf_res**2)
    disparity_net = DisparityNet()

    refine_net_path = os.path.join(args.save_dir, "ckpt", "refine_{}.ckpt".format(args.use_epoch))
    disparity_net_path = os.path.join(args.save_dir, "ckpt", "disp_{}.ckpt".format(args.use_epoch))

    refine_net.load_state_dict(torch.load(refine_net_path))
    disparity_net.load_state_dict(torch.load(disparity_net_path))

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        disparity_net = disparity_net.cuda()
    
    refine_net.eval()
    disparity_net.eval()

    syn_results = []
    target_results = []
    max_psnr = 0
    avg_psnr = 0
    
    loss_to_dist = 0
    for r in range(1, args.runs + 1):
        print("Run ", r)
        psnr_avg = AverageMeter()
        for i, (stereo_pair, target_lf, row_idx, left_idx, right_idx) in enumerate(dataloader):
            n = stereo_pair.shape[0] # batch size
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
            coarse_lf_left, disp1 = synthesize_lf_and_stereo(
                left, right, row_idx, left_idx, stereo_ratio, dataset.lf_res, 
                disparity_net, False, args
            )
            #disp = disp1[0].squeeze().detach().cpu().numpy()
            #disp = normalize(disp) * 255
            #save_image(disp, 'test.png')
            #exit()
            coarse_lf_right, disp2 = synthesize_lf_and_stereo(
                right, left, row_idx, right_idx, stereo_ratio, dataset.lf_res,
                disparity_net, True, args
            )
            disp = disp2[0].squeeze().detach().cpu().numpy()
            disp = normalize(disp) * 255
            save_image(disp, 'test.png')
            #exit()
            merged_lf = (coarse_lf_left + coarse_lf_right) * 0.5
            syn_lf = refine_net(merged_lf)
            

            dist, loss = view_loss_to_dist(
                syn_lf, target_lf, row_idx, left_idx, right_idx,
                dataset.lf_res, F.mse_loss
            )
            loss_to_dist += loss
            syn_lf = syn_lf.detach().cpu().numpy()
            target_lf = target_lf.detach().cpu().numpy()
            psnr = metrics.psnr(syn_lf, target_lf)
            print("PSNR: ", psnr)
            psnr_avg.update(psnr, n)
        
        max_psnr = max(max_psnr, psnr_avg.avg)
        avg_psnr += psnr_avg.avg
            #target_results.append(target_lf)
            #syn_results.append(syn_results)
    avg_psnr /= args.runs
    print("Best PSNR: ", max_psnr)
    print("Average PSNR: ", avg_psnr)

    loss_to_dist /= len(dataset)
    plot_loss_against_dist(dist, loss_to_dist)
        #np.save(os.path.join(args.save_dir, 'syn.npy'), syn_lf)
        #np.save(os.path.join(args.save_dir, 'target.npy'), target_lf)


if __name__ == '__main__':
    run_inference()