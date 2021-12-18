import h5py
import numpy as np
import torch
from image_utils import crop_lf, resize_lf, resize_lf_ratio
from image_utils import sample_lf_index, sample_stereo_index
from image_utils import save_image
from utils import denorm_tanh
from PIL import Image

class LFDataset:
    """ All light field dataset inherits this base dataset """ 

    def __init__(self, root, train, im_size, transform, use_crop=False, mode="stereo"):
        assert mode in ["stereo", "single", "4crop", "2crop"]
        self.mode = mode
        self.root = root
        self.train = train
        self.im_size = im_size
        self.transform = transform
        self.dataset = h5py.File(self.root, 'r')
        self.use_crop = use_crop
        #self.dataset_parts = list(self.dataset.keys())

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
        classname = self.__class__.__name__
        if self.use_crop:
            lf = crop_lf(lf, self.im_size) # (lf_size, lf_size, imsize, imsize, C)
        else:
            lf = resize_lf(lf, self.im_size) #/ 255 # use resize
            
        lf = lf / 255 # scale to [0, 1] range

        lf = lf.reshape(self._lf_size * self._lf_size, *lf.shape[2:])
        if self.transform:
            # apply transformation on light field
            lf = self.transform(lf)
        lf = lf.reshape(self._lf_size, self._lf_size, *lf.shape[1:]) # (N, N, H, W, 3)

        if self.mode == "stereo":
            if self.train:
                # Sample stereo views when training
                #stereo_row_idx, stereo_left_idx, stereo_right_idx = sample_stereo_index(self._lf_size)
                
                stereo_row_idx = self.lf_res // 2 # middle row
                stereo_left_idx = 0 # wide left view
                stereo_right_idx = self.lf_res - 1 # wide right view
            else:
                stereo_row_idx = self.lf_res // 2 # middle row
                stereo_left_idx = 0 # wide left view
                stereo_right_idx = self.lf_res - 1 # wide right view
            
            stereo_left_image = lf[stereo_row_idx, stereo_left_idx]
            stereo_right_image = lf[stereo_row_idx, stereo_right_idx]

            paired_image = torch.cat([stereo_left_image, stereo_right_image], dim=-1)
            target_lf = lf.view(self._lf_size * self._lf_size, *lf.shape[2:])

            return paired_image, target_lf, stereo_row_idx, stereo_left_idx, stereo_right_idx
        elif self.mode == "single":
            center_image = lf[self._lf_size // 2, self._lf_size // 2]
            target_lf = lf.view(self._lf_size * self._lf_size, *lf.shape[2:])
            return center_image, target_lf
        elif self.mode == "4crop": # 4-crops
            # get 4 corner crops
            corner_i = [0, 0, self.lf_res - 1, self.lf_res - 1]
            corner_j = [0, self.lf_res - 1, 0, self.lf_res - 1]
            corner_views = lf[corner_i, corner_j] # (4, h, w, 3)
            if self.train:
                target_i = np.random.randint(self.lf_res)
                target_j = np.random.randint(self.lf_res)
                target_view = lf[target_i, target_j]
                return corner_views, target_view, target_i, target_j
            else:
                return corner_views, lf
        elif self.mode == "2crop":
            corner_i = [self.lf_res // 2, self.lf_res // 2]
            corner_j = [0, self.lf_res - 1]
            corner_views = lf[corner_i, corner_j] # (2, h, w, 3)
            if self.train:
                target_i = np.random.randint(self.lf_res)
                target_j = np.random.randint(self.lf_res)
                target_view = lf[target_i, target_j]
                return corner_views, target_view, target_i, target_j
            else:
                return corner_views, lf

class HCIDataset(LFDataset):
    """ Note: read from h5 file """
    """ Medium-baseline dataset """

    def __init__(self, root, train=True, im_size=256, transform=None, use_all=False, use_crop=False, mode="stereo"):
        super(HCIDataset, self).__init__(root, train, im_size, transform, use_crop, mode)
        self.dataset_parts = list(self.dataset.keys())
        self._lf_size = 9 # lf resolution
        assert set(self.dataset_parts) == set(['test', 'train'])
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

class StanfordDataset(LFDataset):
    """ 
        No testing set, use k-fold cross validation. 
        Manually sample train / test sets.
        11 light fields. Partition into [9, 2] and k-fold
        [0, 1] [2, 3] [4, 5] [6, 7] [8, 9] testing
    """

    def __init__(self, root, train=True, im_size=128, transform=None,  fold=0, use_crop=False, mode="stereo"):
        assert 0 <= fold < 5
        super(StanfordDataset, self).__init__(root, train, im_size, transform, use_crop=use_crop, mode=mode)
        self.root = root
        self.dataset = h5py.File(self.root, 'r')
        self.lf_names = list(self.dataset.keys())  # access lf data by dataset[name][()]
        self._lf_size = 9 # Original 17 * 17. Sample center 9 * 9 for training
        self.lf_names.sort()
        self.fold = fold

        self.test_ids = [2 * fold, 2 * fold + 1]
        self.train_ids = list(range(len(self.lf_names)))
        for idx in self.test_ids:
            self.train_ids.remove(idx)

        if self.train:
            self.lf_names = [self.lf_names[i] for i in self.train_ids]
        else:
            self.lf_names = [self.lf_names[i] for i in self.test_ids]

    def get_single_lf(self, i):
        assert 0 <= i < len(self.lf_names)
        lf = self.dataset[self.lf_names[i]][()] # 17 * 17
        lf = lf[8-4:8+5, 8-4:8+5] # only use 9*9  
        # don't downscale now
        lf = resize_lf(lf, 512) # resize to 512 * 512
        return lf
        
class INRIADataset(LFDataset):
    def __init__(self, root, train=True, im_size=256, transform=None, use_all=False, use_crop=False, mode="stereo"):
        super(INRIADataset, self).__init__(root, train, im_size, transform, use_crop, mode)
        self.dataset_parts = list(self.dataset.keys())
        self._lf_size = 7

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

def test_dataset(dataset, train):
    if dataset == 'hci':
        dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", train=train)
    elif dataset == 'stanford':
        dataset = StanfordDataset(root="../../../mnt/data2/bchao/lf/stanford/dataset.h5", im_size=128)
    elif dataset == 'inria':
        dataset = INRIADataset(root="../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5", train=train)
    #print(len(dataset))
    #for i in range(len(dataset)):
    #    lf = dataset.get_single_lf(i)
    #    view = lf[0, 0]
    #    Image.fromarray(view).save("./imgs/{}.png".format(i))
    lf = dataset.get_single_lf(15)
    lf = lf.reshape(lf.shape[0] * lf.shape[1], *lf.shape[2:])
    for i, view in enumerate(lf):
        Image.fromarray(view).save("./imgs/{}.png".format(i))

if __name__ == "__main__":
    test_dataset("inria", False)
    