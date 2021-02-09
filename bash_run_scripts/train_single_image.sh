#!/bin/bash
cd ..
for f in 0 1 2 3 4
do
python3.7 train_single_image.py --imsize 128 --batch_size 9 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --disparity_scale 40 \
    --recon_loss l1 \
    --gpu_id 1 \
    --c_loss_w 0.005 \
    --tv_loss_w 0.01 \
    --dataset stanford \
    --save_dir experiments_single_image \
    --name v1 \
    --use_crop  \
    --fold $f
done