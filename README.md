# Light field from stereo

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