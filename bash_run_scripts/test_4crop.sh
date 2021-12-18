#!/bin/bash
cd ..

for e in 2000 4000 6000 8000 10000
do
echo $e
python3 test_4crop.py --imsize 512 --batch_size 1 \
    --disparity_levels 100 \
    --scale_disparity 4 \
    --dataset hci \
    --save_dir experiments_4crop \
    --name 2021_1217_0735 \
    --gpu_id 1 \
    --use_epoch $e 
done