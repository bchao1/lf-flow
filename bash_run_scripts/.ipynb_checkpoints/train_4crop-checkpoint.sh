#!/bin/bash
cd ..
python3 train_4crop.py --imsize 135 --batch_size 20 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.0002 \
    --disparity_levels 100 \
    --scale_disparity 4 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset hci \
    --save_dir experiments_2crop \
    --name 2021_1218 \
    --use_crop \
    --mode 2crop

python3 train_4crop.py --imsize 135 --batch_size 20 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.0002 \
    --disparity_levels 100 \
    --scale_disparity 4 \
    --recon_loss l2 \
    --gpu_id 2 \
    --dataset inria \
    --save_dir experiments_2crop \
    --name 2021_1218 \
    --use_crop \
    --mode 2crop

python3 train_4crop.py --imsize 135 --batch_size 12 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.0002 \
    --disparity_levels 100 \
    --scale_disparity 4 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset inria_dlfd \
    --save_dir experiments_2crop \
    --name 2021_1218 \
    --use_crop \
    --mode 2crop