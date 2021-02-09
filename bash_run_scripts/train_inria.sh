#!/bin/bash
cd ..
# === Different merging methods ===
# Only left or only right
python3.7 train_best.py --imsize 133 --batch_size 12 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 15 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 1 \
    --flow_consistency_w 0 \
    --tv_loss_w 0.001 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria \
    --save_dir experiments \
    --name ablation_merge_left \
    --use_crop \
    --merge_method left \
    --refine_model shuffle
# right
python3.7 train_best.py --imsize 133 --batch_size 12 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 15 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 1 \
    --flow_consistency_w 0 \
    --tv_loss_w 0.001 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria \
    --save_dir experiments \
    --name ablation_merge_right \
    --use_crop \
    --merge_method right \
    --refine_model shuffle
# average
python3.7 train_best.py --imsize 133 --batch_size 12 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 15 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 1 \
    --flow_consistency_w 0 \
    --tv_loss_w 0.001 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria \
    --save_dir experiments \
    --name ablation_merge_avg \
    --use_crop \
    --merge_method avg \
    --refine_model shuffle

# === Loss ablation ===
# No consistency
python3.7 train_best.py --imsize 133 --batch_size 12 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 15 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 0 \
    --flow_consistency_w 0 \
    --tv_loss_w 0.001 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria \
    --save_dir experiments \
    --name ablation_no_consistency \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

# No total variation
python3.7 train_best.py --imsize 133 --batch_size 12 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 15 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 1 \
    --flow_consistency_w 0 \
    --tv_loss_w 0 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria \
    --save_dir experiments \
    --name ablation_no_tv \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

# Different model