#!/bin/bash
python3.7 train.py --imsize 128 --batch_size 2 \
    --save_epochs 100 \
    --train_epochs 1000 \
    --lr 0.001 \
    --max_disparity 10 \
    --recon_loss l2 \
    --gpu_id 1 \
    --dataset hci \
    --save_dir experiments \
    --name test_code