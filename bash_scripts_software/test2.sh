cd ..
# due to padding bugs, imsize needs to be:
# - divisible by 7 for INRIA model
# - divisible by 9 for HCI model
# INRIA best: 371
# HCI best: 504
imsize="504" 

# INRIA best: 15 
# HCI   best: 35 
# Can play with this parameter.
max_disparity="35" 

# value: inria or hci
dataset="hci"

python3.6 test.py --imsize $imsize --batch_size 1 \
    --max_disparity $max_disparity \
    --dataset $dataset \
    --save_dir experiments \
    --name deploy \
    --use_epoch 2000 \
    --disp_model original \
    --gpu_id 1 \
    --merge_method alpha \
    --refine_model shuffle \
    --mode stereo