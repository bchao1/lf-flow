#!/bin/bash
cd ..
python3.7 train_4crop.py --imsize 128 --batch_size 8 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --disparity_levels 100 \
    --recon_loss l1 \
    --gpu_id 0 \
    --c_loss_w 0.005 \
    --tv_loss_w 0.01 \
    --dataset hci \
    --save_dir experiments_4crop \
    --name all