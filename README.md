# Light field from stereo

> by Brian Chao


Code for ICIP 2021 and ISMAR/TVCG 2022 paper. 

- [ISMAR/TVCG 2022 Paper link](https://bchao1.github.io/papers/tvcg2022.pdf)
- [ICIP 2021 Paper link](https://bchao1.github.io/papers/icip2021.pdf)


## Datasets
Datasets are saved on server `140.112.21.58`, in `/mnt/data2/bchao/lf/tcsvt_datasets/`:
- `hci/dataset.h5`
- `inria_dlfd/dataset.h5`
- `inria_lytro/dataset.h5`
  
There are a total of 3 datasets. The original dataset sources are:
- [hci](https://lightfield-analysis.uni-konstanz.de)
- [inria_dlfd](http://clim.inria.fr/Datasets/InriaSynLF/index.html)
- [inria_lytro](http://clim.inria.fr/research/LowRank2/datasets/datasets.html)
   
The functions for dataset preprocessing is in `preprocessing/process_dataset.py`. The light field subviews are organized into a `dataset.h5` file for each dataset. You can modify the dataset paths in the `preprocessing/config.yaml` config file.
   
The dataset paths are defined in in function `get_dataset_and_loader()` in `train_best.py`. The dataset class definition is in `lf_datasets.py`. All light field dataset classes, namely:
- `HCIDataset` (hci)
- `INRIADataset` (inria_lytro)
- `INRIA_DLFD_Dataset` (inria_dlfd)
   
inherit the `LFDataset` class. The arguments are:

- `root`: Path to `dataset.h5` file
- `train`: Training or testing dat
- `im_size`: Light field sub-view image size for training or testing. Set to 135 in all training experiments. Set to full image resolution (usually 512) when testing.
- `transform`: Preprocessing function to apply on input light field data
- `use_crop`: If specified, crop image to `im_size`. Otherwise, resize image to `im_size`.
- `mode`: Output data format. 
    - `stereo_wide` or `stereo_narrow`: used in proposed method. For `_wide`, the stereo images extracted from the light field is left and right most subviews from the middle row of the whole light field subview grid. For `_narrow`, the stereo images are the left and right images adjacent to the center view. See [ISMAR/TVCG 2022 paper](https://bchao1.github.io/papers/tvcg2022.pdf) for more info.
    - `single`: used in method proposed by Srinivasan et. al. The center view is returned.
    - `4crop`, `2crop_wide`, `2crop_narrow`: used in method proposed by Kalantari et. al. The stereo images returned in modes `2crop_wide` and `2crop_narrow` are the same as that of `stereo_wide` and `stereo_narrow`, respectively. `4crop` returns the 4 corner views in the light field. Note that the target image in this mode is only a single subview is returned in this mode, unlike whole light field as in `stereo_wide` or `stereo_narrow`. 


## Trained weights
Trained weights are saved on server `140.112.21.58`, in `/mnt/data2/bchao/tvcg_models`, organized by dataset name and experiment name. The directory tree structure is as follows:

- `/mnt/data2/bchao/tvcg_models`
    - `hci`
        - `20211126_1420_final`
            - `ckpt`
                - `disp_1000.ckpt`
                - `refine_1000.ckpt`
                - ...
    - `inria_dlfd`
        - ...


## Training and testing

Main training script: `train_best.py`. Testing script is `test.py`.
See `bash_run_scripts/train.sh` and `bash_run_scripts/test.sh` for more details on how to use arguments.

### Training arguments
- `imsize`: Image size of dataset.
- `save_epochs`: Save model frequency.
- `use_crop`: Crop or resize image to `imsize`. If specified, use cropping rather than resize.
- `max_disparity`: disparity scaling factor for input stereo images. Value depends on datasets.

|Configuration|`max_disparity` value|
|---|---|
|hci|32|
|inria_dlfd|40|
|inria_lyro|10|

- `disp_model`: Disparity network architecture. Default `original`.
- `refine_model`: Refinement network architecture. Default `3dcnn`.
- `refine_hidden`: Refinement network hidden channels. Default 128.
- `merge_method`: Merge method (see paper for more details). Best model uses `alpha`.
- `recon_loss`: Reconstruction loss. Default `l1`.
- `consistency_w`: Weight of left-right consistency loss.
- `tv_loss_w`: Weight of disparity smoothing loss.
- `dataset`: Light field dataset to use. Currently supported: `hci`, `inria_lytro`, `inria_dlfd`. 
- `name`: Name of experiment.
- `mode`: input stereo image configuration `stereo_wide` or `stereo_narrow`. See paper or above for more details on this argument.

### Additional testing arguments
- `test_mode`: Testing mode
    - `normal`: Compute PSNR, SSIM over all testing data
    - `horizontal`: Computer PSNR, SSIM over middle row subviews in light field testing data. To compare with Zhang et. al.
    - `stereo`: Use own stereo images as testing data. 
--- 

# For software team (Eric and Andy)
## Data and weights
- Extract `production.tgz`
- Find model weights (.ckpt files) in extracted `production/weights/hci`, `production/weights/inria` folder 
- Put `disp_2000.ckpt` and `refine_2000.ckpt` model weights in:
    - `experiments/hci/deploy/ckpt` for HCI model
    - `experiments/inria/deploy/ckpt` for INRIA model
- Find testing data in `production/testing_data/hci_testing_data`, `production/testing_data/inria_testing_data` folder
    - Stereo images are labelled by their id (0, 1, 2, ...) and `left`, `right` prefixes
    - During inference, put stereo images in `data/` folder
    - Modify file path in `single_stereo` function in `test.py`

## Inference scripts
- See `bash_scripts_software/`