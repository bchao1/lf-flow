#!/bin/bash
cd ..


for e in 10000
do
echo $e
python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --dataset "hci" \
        --save_dir "experiments_2crop" \
        --name "20220126_1808" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_wide
done
exit
python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --max_disparity 16 \
        --dataset "hci" \
        --save_dir "../results" \
        --name "20211228_2219_narrow" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_narrow

python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --max_disparity 10 \
        --dataset "inria_lytro" \
        --save_dir "../results" \
        --name "20211228_2212_wide" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_wide

python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --max_disparity 5 \
        --dataset "inria_lytro" \
        --save_dir "../results" \
        --name "20211228_2219_narrow" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_narrow

python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --max_disparity 40 \
        --dataset "inria_dlfd" \
        --save_dir "../results" \
        --name "20211228_2212_wide" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_wide

python3 test_4crop.py --imsize 512 --batch_size 1 \
        --disparity_levels 100 \
        --scale_disparity 1 \
        --max_disparity 20 \
        --dataset "inria_dlfd" \
        --save_dir "../results" \
        --name "20211228_2219_narrow" \
        --gpu_id 0 \
        --use_epoch $e \
        --mode 2crop_narrow
done