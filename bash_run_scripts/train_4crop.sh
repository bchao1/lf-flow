#!/bin/bash
cd ..
for f in 0 1 2 3 4
do
python3.7 train_4crop.py --imsize 128 --batch_size 9 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --disparity_levels 100 \
    --scale_disparity 40 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset stanford \
    --save_dir experiments_4crop \
    --name best \
    --use_crop \
    --fold $f
done
