#!/bin/bash
cd ..
python3.7 train_single_image.py --imsize 128 --batch_size 8 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --disparity_scale 4 \
    --recon_loss l1 \
    --gpu_id 0 \
    --c_loss_w 0.005 \
    --tv_loss_w 0.01 \
    --dataset hci \
    --save_dir experiments_single_image \
    --name all