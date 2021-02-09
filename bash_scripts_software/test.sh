
# due to padding bugs, imsize needs to be:
# - divisible by 7 for INRIA model
# - divisible by 9 for HCI model
imsize="371" 

# value:
# - INRIA best: 15, 
# - HCI   best: 35. 
# Can play with this parameter.
max_disparity="15" 

# value: inria or hci
dataset="inria"

python3.7 test.py --imsize $imsize --batch_size 1 \
    --max_disparity $max_disparity \
    --dataset $dataset \
    --save_dir experiments \
    --name recon_l1_best_shuffle \
    --use_epoch 2000 \
    --disp_model original \
    --gpu_id 1 \
    --merge_method alpha \
    --refine_model shuffle \
    --mode stereo