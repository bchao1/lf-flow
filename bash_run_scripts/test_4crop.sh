#!/bin/bash
cd ..
rm -rf ./temp/*

for fold in 0 1 2 3 4
do
python3.7 test_4crop.py --imsize 512 --batch_size 1 \
    --disparity_levels 100 \
    --scale_disparity 40 \
    --dataset stanford \
    --save_dir experiments_4crop \
    --name best \
    --gpu_id 1 \
    --use_epoch 2000 \
    --fold $fold
done