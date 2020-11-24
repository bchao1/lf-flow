#!/bin/bash
cd ..
echo "Train with alpha blending ==>"
python3.7 train.py --imsize 128 --batch_size 8 \
    --save_epochs 100 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 10 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --recon_loss l2 \
    --use_alpha_blending \
    --gpu_id 1 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l2_alpha

echo "Train with weighted views ==>"
python3.7 train.py --imsize 128 --batch_size 8 \
    --save_epochs 100 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 10 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --recon_loss l2 \
    --use_weighted_view \
    --gpu_id 1 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l2_weighted_view