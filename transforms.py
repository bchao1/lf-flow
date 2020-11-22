import torch

class ToTensor:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x)

class Normalize01:
    """ Perform channel-wise [0, 1] normalization on [H, W, C] tensor """

    def __init__(self):
        pass
    
    def __call__(self, x):
        c = x.shape[-1]
        channel_min = torch.min(x.reshape(-1, c), dim=0)[0].reshape(1, 1, c)
        channel_max = torch.max(x.reshape(-1, c), dim=0)[0].reshape(1, 1, c)
        x = (x - channel_min) / (channel_max - channel_min + 1e-10)
        return x

class NormalizeRange:
    """ Perform channel-wise [a, b] normalization on [H, W, C] tensor """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.norm01 = Normalize01()
    
    def __call__(self, x):
        #x = self.norm01(x)
        x = (self.high - self.low) * x + self.low 
        return x

if __name__ == '__main__':
    x = torch.randn(5, 5, 3)
    t = NormalizeRange(-1, 1)
    o = t(x)
    print(o[:, :, 0])