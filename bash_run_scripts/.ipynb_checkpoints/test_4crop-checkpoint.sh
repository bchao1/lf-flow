#!/bin/bash
cd ..

for dataset in "hci" "inria" "inria_dlfd"
do
    echo $dataset
    for e in 2000 4000 6000 8000 10000
    do
    echo $e
    python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 4 \
        --dataset $dataset \
        --save_dir experiments_2crop \
        --name 2021_1218 \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop
    done
done