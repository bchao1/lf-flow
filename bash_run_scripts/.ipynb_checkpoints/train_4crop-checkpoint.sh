#!/bin/bash
cd ..
python3 train_4crop.py --imsize 135 --batch_size 20 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --disparity_levels 100 \
    --scale_disparity 1 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset hci \
    --save_dir /results \
    --name 20211228_2200 \
    --use_crop \
    --mode 2crop_narrow

python3 train_4crop.py --imsize 135 --batch_size 20 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --disparity_levels 100 \
    --scale_disparity 1 \
    --recon_loss l2 \
    --gpu_id 2 \
    --dataset inria_lytro \
    --save_dir /results \
    --name 20211228_2200 \
    --use_crop \
    --mode 2crop_narrow

python3 train_4crop.py --imsize 135 --batch_size 12 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --disparity_levels 100 \
    --scale_disparity 1 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2200 \
    --use_crop \
    --mode 2crop_narrow