#!/bin/bash
cd ..
python3.7 train.py --imsize 128 --batch_size 8 \
    --save_epochs 100 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 10 \
    --edge_loss l2 \
    --edge_loss_w 0.2 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --recon_loss l2 \
    --gpu_id 0 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l2_edge_l2

python3.7 train.py --imsize 128 --batch_size 8 \
    --save_epochs 100 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 10 \
    --edge_loss l1 \
    --edge_loss_w 0.2 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --recon_loss l2 \
    --gpu_id 0 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l2_edge_l1