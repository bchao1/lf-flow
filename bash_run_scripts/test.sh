#!/bin/bash
cd ..
e="2000"
echo "Using $e ... "
echo "Baseline"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 20 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l1_256 \
    --use_epoch $e
echo "With edge"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 20 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l1_edge_l2_256 \
    --use_epoch $e
echo "With alpha"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 20 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l1_alpha_256 \
    --use_epoch $e
echo "With weighted"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 20 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l1_weighted_view_256 \
    --use_epoch $e
echo "All"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 20 \
    --dataset hci \
    --save_dir experiments \
    --name all \
    --use_epoch $e