import os
import numpy as np
from PIL import Image
import yaml
import h5py
import time

def preprocess_lf_dataset(name):
    with open('dataset_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if name == 'hci':
        preprocess_hci_dataset(root=config['hci'])
    elif name == 'stanford':
        preprocess_stanford_dataset(root=config['stanford'])
    elif name == 'inria':
        preprocess_inria_dataset(root=config['inria'])
    else:
        raise Error()

def preprocess_hci_dataset(root):
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
        imgs = imgs.reshape(9, 9, *imgs.shape[1:])
        return imgs

    use = ['training', 'test', 'additional']
    for subdir in use:
        print("Processing {} ...".format(subdir))
        group = file.create_group(subdir)
        folder = os.path.join(root, subdir)
        lfs = os.listdir(folder)
        lfs = [lf for lf in lfs if not lf.endswith('.txt')]  # all light field folders
        for lf in lfs:
            print("Processing {} ...".format(lf))
            data = process_lf_folder(os.path.join(folder, lf))
            dset = group.create_dataset(lf, data=data)

    file.close()

def preprocess_stanford_dataset(root):
    os.remove(os.path.join(root, 'dataset.h5'))
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'w')

    def process_lf_folder(lf_folder):
        img_files = os.listdir(lf_folder)
        if 'dataset.h5' in img_files:
            img_files.remove('dataset.h5')
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
        print("Processing {} ...".format(folder))
        data = process_lf_folder(os.path.join(root, folder, 'rectified'))
        dset = file.create_dataset(folder, data=data)
    file.close()

def test_dataset_h5(name):
    with open('dataset_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if name == 'hci':
        test_hci_dataset(root=config['hci'])
    elif name == 'stanford':
        test_stanford_dataset(root=config['stanford'])
    elif name == 'inria':
        test_inria_dataset(root=config['inria'])
    else:
        raise Error()

def test_hci_dataset(root):
    # Access h5 file
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    data = file['training']['boxes'][()]
    print("Access h5 file time : {}".format(time.time() - start))

def test_stanford_dataset(root):
    # Access h5 file
    start = time.time()
    file = h5py.File(os.path.join(root, 'dataset.h5'), 'r')
    data = file['beans'][()]
    print("Lf size: {}".format(data.shape))
    print("Access h5 file time : {}".format(time.time() - start))

def test_inria_dataset(root):
    pass

if __name__ == '__main__':
    #preprocess_lf_dataset('stanford')
    test_dataset_h5('stanford')
