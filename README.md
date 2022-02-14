# Light field from stereo

## 2022/2/4 update

[Paper link](https://bchao1.github.io/papers/tvcg2022.pdf).


### Datasets and trained weights
Light field datasets are in `/mnt/data2/bchao/lf/tcsvt_datasets/`:
- `hci/dataset.h5`
- `inria_dlfd/dataset.h5`
- `inria_lytro/dataset.h5`

Define the dataset path in function `get_dataset_and_loader()` in `train_best.py`. The dataset class definition is in `lf_datasets.py`.
   
Trained weights are in `/mnt/data2/bchao/tvcg_models`, organized by dataset name and experiment name. The directory tree structure is as follows:

- `/mnt/data2/bchao/tvcg_models`
    - `hci`
        - `20211126_1420_final`
            - `ckpt`
                - `disp_1000.ckpt`
                - `refine_1000.ckpt`
                - ...
    - `inria_dlfd`
        - ...


### Training and testing

Main training script: `train_best.py`. Testing script is `test.py`.
See `bash_run_scripts/train.sh` and `bash_run_scripts/test.sh` for more details on how to use arguments.

### Arguments
- `mode`: input stereo image configuration `stereo_wide` or `stereo_narrow`. See paper for more details on this argument.
- `max_disparity`: disparity scaling factor for input stereo images. Value depends on datasets.

|Configuration|`max_disparity` value|
|---|---|
|hci|32|
|inria_dlfd|40|
|inria_lyro|10|




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