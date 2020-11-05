import h5py

class HCIDataset:
    """ Note: read from h5 file """
    """ Medium-baseline dataset """

    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.dataset = h5py.File(self.root, 'r')
        self.dataset_parts = list(self.dataset.keys())
        assert set(self.dataset_parts) == set(['additional', 'test', 'training'])
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
    def num_lfs(self):
        return len(self.lf_names)

    def get_single_lf(self, i):
        assert 0 <= i < len(self.lf_names)
        return self.dataset[self.lf_names[i]]
    
    def __getitem__(self, i):
        pass

class StanfordDataset:
    """ No testing set, use k-fold cross validation. """

    def __init__(self, root):
        self.root = root
        self.dataset = h5py.File(self.root, 'r')
        self.lf_names = list(self.dataset.keys())  # access lf data by dataset[name][()]
    
    def __getitem__(self, i):
        pass
        
class INRIADataset:
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.dataset = h5py.File(self.root, 'r')
        if self.train:
            self.dataset = self.dataset['Training']
        else:
            self.dataset = self.dataset['Testing']
        self.lf_names = list(self.dataset.keys()) # access lf data by dataset[name][()]
    
    def __getitem__(self, i):
        pass

        

if __name__ == "__main__":
    #dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5")
    #dataset = StanfordDataset(root="../../../mnt/data2/bchao/lf/stanford/dataset.h5")
    #dataset = INRIADataset(root="../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5")
    pass