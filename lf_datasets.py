import h5py
import numpy as np
import torch
from image_utils import crop_lf, resize_lf
from image_utils import sample_lf_index, sample_stereo_index

class HCIDataset:
    """ Note: read from h5 file """
    """ Medium-baseline dataset """

    def __init__(self, root, train=True, im_size=256, transform=None, use_all=False):
        self.root = root
        self.train = train
        self.dataset = h5py.File(self.root, 'r')
        self.dataset_parts = list(self.dataset.keys())
        self.im_size = im_size
        self.transform = transform
        self._lf_size = 9 # lf resolution
        assert set(self.dataset_parts) == set(['additional', 'test', 'training'])
        if not use_all:
            if self.train:
                self.dataset_parts.remove('test')  # use 'additional' and 'training' as training data
            else:
                self.dataset_parts = ['test']
        self.lf_names = []  # access lf data by dataset[name][()]
        for part in self.dataset_parts:
            lfs = list(self.dataset[part].keys())
            self.lf_names.extend(['/'.join([part, lf]) for lf in lfs])
        self.lf_names.sort()

    @property
    def lf_res(self):
        return self._lf_size

    @property
    def num_lfs(self):
        return len(self.lf_names)

    def get_single_lf(self, i):
        # get full original resolution lf
        assert 0 <= i < len(self.lf_names)
        return self.dataset[self.lf_names[i]][()]
    
    def __len__(self):
        return len(self.lf_names)

    def __getitem__(self, i):
        lf = self.get_single_lf(i) # retrieve original lf
        #cropped_lf = crop_lf(lf, self.im_size) # (lf_size, lf_size, imsize, imsize, C)
        lf = resize_lf(lf, self.im_size) / 255 # use resize

        stereo_row_idx, stereo_left_idx, stereo_right_idx = sample_stereo_index(self._lf_size)
        #view_row_idx, view_col_idx = sample_lf_index(self._lf_size)
        
        #row_shear = view_row_idx - stereo_row_idx
        #left_shear = view_col_idx - stereo_left_idx
        #right_shear = view_col_idx - stereo_right_idx

        #target_lf_image = cropped_lf[view_row_idx, view_col_idx]          / 255
        stereo_left_image = lf[stereo_row_idx, stereo_left_idx]
        stereo_right_image = lf[stereo_row_idx, stereo_right_idx]

        paired_image = np.concatenate([stereo_left_image, stereo_right_image], axis=-1)
        target_lf = lf.reshape(self._lf_size * self._lf_size, *lf.shape[2:])

        if self.transform:
            paired_image = self.transform(paired_image) # [N, H, W, 2C]
            target_lf = self.transform(target_lf) # [N, U*V, H, W, C]
            #stereo_left_idx = torch.tensor(stereo_left_idx)
            #stereo_right_idx = torch.tensor(stereo_right_idx)
        
        return paired_image, target_lf, stereo_left_idx, stereo_right_idx

class StanfordDataset:
    """ No testing set, use k-fold cross validation. """

    def __init__(self, root):
        self.root = root
        self.dataset = h5py.File(self.root, 'r')
        self.lf_names = list(self.dataset.keys())  # access lf data by dataset[name][()]
    
    @property
    def num_lfs(self):
        return len(self.lf_names)

    def get_single_lf(self, i):
        assert 0 <= i < len(self.lf_names)
        return self.dataset[self.lf_names[i]][()]
    
    def __getitem__(self, i):
        pass
        
class INRIADataset:
    def __init__(self, root, train=True, use_all=False):
        self.root = root
        self.train = train
        self.dataset = h5py.File(self.root, 'r')
        self.dataset_parts = list(self.dataset.keys())
        assert set(self.dataset_parts) == set(['Training', 'Testing'])
        if not use_all:
            if self.train:
                self.dataset_parts = ['Training']
            else:
                self.dataset_parts = ['Testing']
        self.lf_names = []  # access lf data by dataset[name][()]
        for part in self.dataset_parts:
            lfs = list(self.dataset[part].keys())
            self.lf_names.extend(['/'.join([part, lf]) for lf in lfs])
        self.lf_names.sort()
    
    @property
    def num_lfs(self):
        return len(self.lf_names)

    def get_single_lf(self, i):
        assert 0 <= i < len(self.lf_names)
        return self.dataset[self.lf_names[i]]
    
    def __getitem__(self, i):
        pass

if __name__ == "__main__":
    #dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5")
    #dataset = StanfordDataset(root="../../../mnt/data2/bchao/lf/stanford/dataset.h5")
    #dataset = INRIADataset(root="../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5")
    pass