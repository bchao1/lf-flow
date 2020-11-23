#!/bin/bash
python3.7 test.py --imsize 256 --batch_size 1 \
    --max_disparity 10 \
    --dataset hci \
    --save_dir experiments \
    --runs 10 \
    --name "1123"