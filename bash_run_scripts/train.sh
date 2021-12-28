#!/bin/bash
cd ..

# DLFD (4+1) * 8 =40
python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_merge_left \
    --use_crop \
    --merge_method left \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 1 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_merge_right \
    --use_crop \
    --merge_method right \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 2 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_merge_avg \
    --use_crop \
    --merge_method avg \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 3 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_3dcnn \
    --use_crop \
    --merge_method alpha \
    --refine_model 3dcnn


python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 0 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 4 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_nolr \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0 \
    --recon_loss l1 \
    --gpu_id 5 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_tv_0 \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.1 \
    --recon_loss l1 \
    --gpu_id 6 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_tv_0.1 \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 40 \
    --consistency_w 1 \
    --tv_loss_w 0.01 \
    --recon_loss l1 \
    --gpu_id 7 \
    --dataset inria_dlfd \
    --save_dir /results \
    --name 20211228_2100_tv_0.01 \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

    
exit
# HCI max disparity = 4 * 8 = 32

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 0 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_final \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle
    
exit
# no LR
python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 0.0 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 1 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_nolr \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle
    
#
python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 2 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_3dcnn \
    --use_crop \
    --merge_method alpha \
    --refine_model 3dcnn
    

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 2 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_merge_alpha \
    --use_crop \
    --merge_method alpha \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 3 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_merge_left \
    --use_crop \
    --merge_method left \
    --refine_model shuffle
    
python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 4 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_merge_right \
    --use_crop \
    --merge_method right \
    --refine_model shuffle

python3 train_best.py --imsize 135 --batch_size 10 \
    --save_epochs 1000 \
    --train_epochs 10000 \
    --lr 0.001 \
    --max_disparity 32 \
    --consistency_w 1 \
    --tv_loss_w 0.001 \
    --recon_loss l1 \
    --gpu_id 5 \
    --dataset hci \
    --save_dir /results \
    --name 20211226_1420_merge_avg \
    --use_crop \
    --merge_method avg \
    --refine_model shuffle
    
exit