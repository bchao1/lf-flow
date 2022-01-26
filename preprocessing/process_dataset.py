import os
import numpy as np
from PIL import Image
import yaml
import h5py
import time
from utils import read_pfm
import glob

def preprocess_lf_dataset(name):
    with open('dataset_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if name == 'hci':
        preprocess_hci_dataset(root=config['hci'])
    elif name == 'stanford':
        preprocess_stanford_dataset(root=config['stanford'])
    elif name == 'inria':
        preprocess_inria_dataset(root=config['inria'])
    elif name == 'inria_dlfd':
        preprocess_inria_dlfd_dataset(root=config['inria_dlfd'])
    else:
        raise Error()

def preprocess_hci_dataset(root):
    if os.path.exists(os.path.join(root, 'dataset.h5')):
        os.remove(os.path.join(root, 'dataset.h5'))
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'w')

    def process_lf_folder(lf_folder):
        
        img_files = os.listdir(lf_folder)
        img_files = [file for file in img_files if file.startswith('input_Cam')]
        img_files.sort()
        assert len(img_files) == 81  # 9*9 light field
        imgs = [Image.open(os.path.join(lf_folder, path)) for path in img_files]
        imgs = [np.array(img) for img in imgs]
        imgs = np.stack(imgs)
        lf = imgs.reshape(9, 9, *imgs.shape[1:])
        return lf

    lf_folders = os.listdir(root)
    lf_folders = [folder for folder in lf_folders if os.path.isdir(os.path.join(root, folder)) ]
    lf_folders.sort()
    print("Total lfs:", len(lf_folders))

    test_lfs = ["bedroom", "bicycle", "herbs", "origami"]
    train_lfs = [folder for folder in lf_folders if folder not in test_lfs] 
    train_group = file.create_group("train")
    test_group = file.create_group("test")
    for lf_folder in lf_folders:
        print(lf_folder)
        data = process_lf_folder(os.path.join(root, lf_folder))
        if lf_folder in train_lfs:
            dset = train_group.create_dataset(lf_folder, data=data)
        else:
            dset = test_group.create_dataset(lf_folder, data=data)
    file.close()

def preprocess_stanford_dataset(root):
    if os.path.exists(os.path.join(root, 'dataset.h5')):
        os.remove(os.path.join(root, 'dataset.h5'))
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'w')

    def process_lf_folder(lf_folder):
        img_files = os.listdir(lf_folder)
        img_files.sort()
        assert len(img_files) == 17**2
        imgs = [Image.open(os.path.join(lf_folder, path)) for path in img_files]
        imgs = [np.array(img) for img in imgs]
        imgs = np.stack(imgs)
        imgs = imgs.reshape(17, 17, *imgs.shape[1:])
        return imgs

    lf_folders = os.listdir(root)
    lf_folders = [f for f in lf_folders if not f.endswith('.zip')]
    for folder in lf_folders:
        if folder.endswith('.h5'):
            continue
        print("Processing {} ...".format(folder))
        data = process_lf_folder(os.path.join(root, folder, 'rectified'))
        dset = file.create_dataset(folder, data=data)
    file.close()

def preprocess_inria_dlfd_dataset(root):
    if os.path.exists(os.path.join(root, 'dataset.h5')):
        os.remove(os.path.join(root, 'dataset.h5'))
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'w')

    def process_lf_folder(lf_folder):
        img_files = glob.glob(lf_folder+"/*.png")
        img_files.sort()
        assert len(img_files) == 9**2
        imgs = [Image.open(os.path.join(lf_folder, path)) for path in img_files]
        imgs = [np.array(img) for img in imgs]
        imgs = np.stack(imgs)
        imgs = imgs.reshape(9, 9, *imgs.shape[1:])
        return imgs
    
    lf_folders = os.listdir(root)
    lf_folders = [folder for folder in lf_folders if os.path.isdir(os.path.join(root, folder)) ]
    lf_folders.sort()
    print("Total lfs:", len(lf_folders))
    test_lfs = lf_folders[:7] # split into 7 test data 
    train_lfs = lf_folders[7:] # 32 training data
    train_group = file.create_group("train")
    test_group = file.create_group("test")
    for lf_folder in lf_folders:
        print(lf_folder)
        data = process_lf_folder(os.path.join(root, lf_folder))
        if lf_folder in train_lfs:
            dset = train_group.create_dataset(lf_folder, data=data)
        else:
            dset = test_group.create_dataset(lf_folder, data=data)
    file.close()

def preprocess_inria_dataset(root):
    if os.path.exists(os.path.join(root, 'dataset.h5')):
        os.remove(os.path.join(root, 'dataset.h5'))
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'w')

    def process_lf_folder(lf_folder):
        img_files = os.listdir(lf_folder)
        img_files.sort()
        assert len(img_files) == 7**2
        imgs = [Image.open(os.path.join(lf_folder, path)) for path in img_files]
        imgs = [np.array(img) for img in imgs]
        imgs = np.stack(imgs)
        imgs = imgs.reshape(7, 7, *imgs.shape[1:])
        return imgs

    use = ['Training', 'Testing']
    for subdir in use:
        print("Processing {} ...".format(subdir))
        group = file.create_group(subdir)
        folder = os.path.join(root, subdir)
        lfs = os.listdir(folder)  # all light field folders
        for lf in lfs:
            if lf in ['.DS_Store']:
                continue
            print("Processing {} ...".format(lf))
            data = process_lf_folder(os.path.join(folder, lf))
            dset = group.create_dataset(lf, data=data)
    file.close()

def test_dataset_h5(name):
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    if name == 'hci':
        test_hci_dataset(root=config['hci'])
    elif name == 'stanford':
        test_stanford_dataset(root=config['stanford'])
    elif name == 'inria_lytro':
        test_inria_lytro_dataset(root=config['inria_lytro'])
    elif name == "inria_dlfd":
        test_inria_dlfd_dataset(root=config['inria_dlfd'])
    else:
        raise Error()

def test_hci_dataset(root):
    # Access h5 file
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    data = file['test']['bicycle'][()]
    print("Access h5 file time : {}".format(time.time() - start))
    data = data[::3, ::3].reshape(-1, *data.shape[2:])
    for i, img in enumerate(data):
        Image.fromarray(img).save(f"../data/interview/{i}.png")
    

def test_stanford_dataset(root):
    # Access h5 file
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    data = file['beans'][()]
    print("Lf size: {}".format(data.shape))
    print("Access h5 file time : {}".format(time.time() - start))

def test_inria_lytro_dataset(root):
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    print(file.keys())
    data = file['Training']['Guitar1'][()]
    print("Lf size: {}".format(data.shape))
    print("Access h5 file time : {}".format(time.time() - start))

def test_inria_dlfd_dataset(root):
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    print(file.keys())
    data = file['train']['Robots_dense'][()]
    print("Lf size: {}".format(data.shape))
    print("Access h5 file time : {}".format(time.time() - start))

def save_stereo(save_root, dataset_name, mode="small"):
    assert mode in ["small", "large"] # stereo mode

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs(os.path.join(save_root, dataset_name), exist_ok=True)
    file = h5py.File(os.path.join(config[dataset_name], 'dataset.h5'), 'r')

    if dataset_name == "hci" or dataset_name == "inria_dlfd":
        test_data = file["test"]
    elif dataset_name == "inria_lytro":
        test_data = file["Testing"]

    for i, lf_name in enumerate(list(test_data.keys())):
        data = test_data[lf_name][()]
        lf_res = data.shape[0]
        if mode == "small":
            left = data[lf_res // 2, lf_res // 2 - 1]
            right = data[lf_res // 2, lf_res // 2 + 1]
        elif mode == "large":
            left = data[lf_res // 2, 0]
            right = data[lf_res // 2, lf_res - 1]
        Image.fromarray(left).save(os.path.join(save_root, dataset_name, f"{i}_left.png"))
        Image.fromarray(right).save(os.path.join(save_root, dataset_name, f"{i}_right.png"))

        


if __name__ == '__main__':
    #preprocess_lf_dataset('hci')
    test_dataset_h5('hci')
    exit()
    save_stereo("../data/zhang_small_baseline_data", "inria_dlfd", "small")
    save_stereo("../data/zhang_small_baseline_data", "inria_lytro", "small")
    save_stereo("../data/zhang_small_baseline_data", "hci", "small")
    save_stereo("../data/zhang_large_baseline_data", "inria_dlfd", "large")
    save_stereo("../data/zhang_large_baseline_data", "inria_lytro", "large")
    save_stereo("../data/zhang_large_baseline_data", "hci", "large")
