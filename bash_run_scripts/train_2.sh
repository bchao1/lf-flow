#!/bin/bash
cd ..
python3.7 train.py --imsize 256 --batch_size 2 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 20 \
    --consistency_w 0 \
    --tv_loss_w 0 \
    --edge_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 1 \
    --dataset hci \
    --save_dir experiments \
    --name no_lr_consistency