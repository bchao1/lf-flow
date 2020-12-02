#!/bin/bash
cd ..
for fold in 3 4
do
python3.7 train.py --imsize 256 --batch_size 4 \
    --save_epochs 500 \
    --train_epochs 2000 \
    --lr 0.001 \
    --max_disparity 50 \
    --edge_loss l2 \
    --edge_loss_w 0 \
    --color_loss_w 0 \
    --consistency_w 1 \
    --recon_loss l1 \
    --gpu_id 1 \
    --dataset stanford \
    --save_dir experiments \
    --name recon_l1_256 \
    --fold $fold
done
