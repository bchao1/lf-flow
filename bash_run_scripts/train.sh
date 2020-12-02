#!/bin/bash
cd ..
python3.7 train.py --imsize 256 --batch_size 2 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 20 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --use_alpha_blending \
    --use_weighted_view \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset hci \
    --save_dir experiments \
    --name all