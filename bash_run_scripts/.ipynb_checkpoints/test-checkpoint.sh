#!/bin/bash
cd ..

imsize="504"
max_disparity="4"
dataset="hci"
save_dir="./experiments/"
name="20211219_1040"
mode="normal"

for e in 2000 4000 6000 8000 10000
do
python3 test.py --imsize $imsize --batch_size 1 \
    --max_disparity $max_disparity \
    --dataset $dataset \
    --save_dir $save_dir \
    --name $name \
    --use_epoch $e \
    --disp_model original \
    --gpu_id 3 \
    --merge_method alpha \
    --refine_model shuffle \
    --mode $mode 
done

# hci and inria
#python3.7 test.py --imsize $imsize --batch_size 1 \
#    --max_disparity $max_disparity \
#    --dataset $dataset \
#    --save_dir experiments \
#    --name recon_l1_best_shuffle \
#    --use_epoch 2000 \
#    --disp_model original \
#    --gpu_id 1 \
#    --merge_method alpha \
#    --refine_model shuffle \
#    --mode normal 

# Stanford
# for fold in 0
# do
# rm -rf ./temp/*
# python3.7 test.py --imsize $imsize --batch_size 1 \
#     --max_disparity $max_disparity \
#     --dataset $dataset \
#     --save_dir experiments \
#     --name best \
#     --use_epoch 2000 \
#     --disp_model original \
#     --gpu_id 1 \
#     --merge_method alpha \
#     --refine_model shuffle \
#     --mode data \
#     --fold $fold
# done

#for m in "avg" "left" "right"
#do
#echo $m
#rm -rf ./temp/*
#python3.7 test.py --imsize $imsize --batch_size 1 \
#    --max_disparity $max_disparity \
#    --dataset $dataset \
#    --save_dir experiments \
#    --name ablation_merge_$m \
#    --use_epoch 2000 \
#    --disp_model original \
#    --gpu_id 1 \
#    --merge_method $m \
#    --refine_model shuffle \
#    --mode normal
#done
#
#for m in "consistency" "tv"
#do
#rm -rf ./temp/*
#echo $m
#python3.7 test.py --imsize $imsize --batch_size 1 \
#    --max_disparity $max_disparity \
#    --dataset $dataset \
#    --save_dir experiments \
#    --name ablation_no_$m \
#    --use_epoch 2000 \
#    --disp_model original \
#    --gpu_id 1 \
#    --merge_method alpha \
#    --refine_model shuffle \
#    --mode normal
#done
#
#rm -rf ./temp/*
#echo "baseline refine"
#python3.7 test.py --imsize $imsize --batch_size 1 \
#    --max_disparity $max_disparity \
#    --dataset $dataset \
#    --save_dir experiments \
#    --name recon_l1_best_consistency_wide_baseline \
#    --use_epoch 2000 \
#    --disp_model original \
#    --gpu_id 1 \
#    --merge_method alpha \
#    --refine_model 3dcnn \
#    --mode normal
#
# === Test stanford ===
#for f in 0 1 2 3 4
#do
#rm ./temp/*
#python3.7 test.py --imsize $imsize --batch_size 1 \
#    --max_disparity $max_disparity \
#    --dataset $dataset \
#    --save_dir experiments \
#    --name best \
#    --use_epoch 2000 \
#    --disp_model original \
#    --gpu_id 1 \
#    --merge_method alpha \
#    --refine_model shuffle \
#    --mode normal \
#    --fold $f
#done