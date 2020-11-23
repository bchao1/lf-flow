import os
import numpy as np
from argparse import ArgumentParser

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
    parser.add_argument("--runs", type=int, default=5) # average result over 5 inference runs

    args = parser.parse_args()
    args_dict = vars(args)
    
    if None in list(args_dict.values()):
        not_specified = [key for key in args_dict if args_dict[key] is None]
        raise ValueError("Please specify: {}".format(", ".join(not_specified)))
    args.save_dir = os.path.join(args.save_dir, args.name)


    # Prepare datasets
    dataset, dataloader = get_dataset_and_loader(args, train=False)

    refine_net = LFRefineNet(in_channels=dataset.lf_res**2)
    disparity_net = DisparityNet()

    refine_net_path = os.path.join(args.save_dir, "refine_{}.ckpt".format(args.use_epoch))
    disparity_net_path = os.path.join(args.save_dir, "disp_{}.ckpt".format(args.use_epoch))

    refine_net.load_state_dict(torch.load(refine_net_path))
    disparity_net.load_state_dict(torch.load(disparity_net_path))

    if torch.cuda.is_available():
        refine_net = refine_net.cuda()
        disparity_net = disparity_net.cuda()
    
    refine_net.eval()
    disparity_net.eval()

    syn_results = []
    target_results = []
    psnr_avg = AverageMeter()
    
    for r in range(1, args.runs + 1):
        print("Run ", r)
        for i, (stereo_pair, target_lf, left_idx, right_idx) in enumerate(dataloader):
            n = stereo_pair.shape[0] # batch size

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
                disparity_net, False, args
            )
            #print(disp1.shape)
            #exit()
            coarse_lf_right, disp2 = synthesize_lf_and_stereo(
                right, left, right_idx, stereo_ratio, dataset.lf_res,
                disparity_net, True, args
            )
            merged_lf = (coarse_lf_left + coarse_lf_right) * 0.5
            syn_lf = refine_net(merged_lf)
            
            syn_lf = syn_lf.detach().cpu().numpy()
            target_lf = target_lf.detach().cpu().numpy()
            
            psnr = metrics.psnr(syn_lf, target_lf)
            print("PSNR: ", psnr)
            psnr_avg.update(psnr, n)

            #target_results.append(target_lf)
            #syn_results.append(syn_results)
    print("PSNR Average: ", psnr_avg.avg)
        #np.save(os.path.join(args.save_dir, 'syn.npy'), syn_lf)
        #np.save(os.path.join(args.save_dir, 'target.npy'), target_lf)


if __name__ == '__main__':
    run_inference()