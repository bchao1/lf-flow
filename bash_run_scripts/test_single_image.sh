#!/bin/bash
cd ..
rm -rf ./temp/*
for fold in 0 1 2 3 4
do
python3.7 test_single_image.py --imsize 512 --batch_size 1 \
    --disparity_scale 40 \
    --dataset stanford \
    --save_dir experiments_single_image \
    --name v1 \
    --gpu_id 0 \
    --use_epoch 2000 \
    --fold $fold
done