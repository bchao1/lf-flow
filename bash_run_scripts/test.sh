#!/bin/bash
echo "Recon l1 ===>"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 10 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l1 

echo "Recon l2 ===>"
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 10 \
    --dataset hci \
    --save_dir experiments \
    --name recon_l2
