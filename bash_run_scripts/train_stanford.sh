#!/bin/bash
cd ..
for fold in 0 1 2 3 4
do
python3.7 train_best.py --imsize 135 --batch_size 9 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 90 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --consistency_w 1 \
    --flow_consistency_w 0 \
    --tv_loss_w 0.001 \
    --rot_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset stanford \
    --save_dir experiments \
    --name best \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle \
    --fold $fold
done