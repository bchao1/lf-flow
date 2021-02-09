import torch
import cv2

class ToTensor:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x)

class Normalize01:
    """ Perform channel-wise [0, 1] normalization on [*, H, W, C] tensor """

    def __init__(self):
        pass
    
    def __call__(self, x):
        c = x.shape[-1]
        if len(x.shape) == 3: # normal data (H, W, C)
            channel_min = torch.min(x.reshape(-1, c), dim=0)[0].reshape(1, 1, c)
            channel_max = torch.max(x.reshape(-1, c), dim=0)[0].reshape(1, 1, c)
        elif len(x.shape) == 4: # light field (N, H, W, C) --> input shape!
            n = x.shape[0]
            channel_min = torch.min(x.reshape(n, -1, c), dim=1)[0].reshape(n, 1, 1, c)
            channel_max = torch.max(x.reshape(n, -1, c), dim=1)[0].reshape(n, 1, 1, c)
        x = (x - channel_min) / (channel_max - channel_min + 1e-10)
        return x

class NormalizeRange:
    """ Perform channel-wise [a, b] normalization on [H, W, C] tensor """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.norm01 = Normalize01()
    
    def __call__(self, x):
        x = self.norm01(x)
        x = (self.high - self.low) * x + self.low 
        return x

# Batch Random Color Jittering on Whole Light Field
# Note: data is in [0, 1] range but not necessarily min = 0, max = 1
class RandomBrightness:
    def __init__(self):
        pass
    
    def __call__(self, x):
        # x: shape(n, h, w, 3). All views must be adjusted with same value.
        # adjust_val [-0.5, 0.5]
        adjust_val = torch.rand(1, 1, 1, 1, dtype=x.dtype).to(x.device)
        x = x + (adjust_val - 0.5)
        x = torch.clamp(x, 0, 1)
        return x

class RandomSaturation:
    def __init__(self):
        #x = torch.randn(5, 3, 64, 64)
        #m = x.mean(dim=[1, 2, 3], keepdim=True)
        #print(m.shape)
        #exit()
        pass 
    
    def __call__(self, x):
        color_mean = x.mean(dim=-1, keepdim=True) # (N, H, W, 1) -> grayscale of each view
        adjust_val = torch.rand(1, 1, 1, 1, dtype=x.dtype).to(x.device)
        x = (x - color_mean) * (adjust_val * 2) + color_mean
        x = torch.clamp(x, 0, 1)
        return x

class RandomContrast:
    def __init__(self):
        pass
    
    def __call__(self, x):
        channel_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        adjust_val = torch.rand(1, 1, 1, 1, dtype=x.dtype).to(x.device)
        x = (x - channel_mean) * (adjust_val + 0.5) + channel_mean
        x = torch.clamp(x, 0, 1)
        return x 

class CenterCrop():
    # center crop resize tensor
    def __init__(self):
        pass
    
    def __call__(self, img):
        # Center crop and resize an image
        c, h, w = img.shape
        if h > w:
            top_pad = (h - w) // 2
            img = img[:, top_pad:top_pad+w, :]
        else:
            left_pad = (w - h) // 2
            img = img[:, :, left_pad:left_pad+h]
        return img

if __name__ == '__main__':
    x = torch.randn(81, 5, 5, 3)
    t = NormalizeRange(-1, 1)
    o = t(x)
    print(o[0][:, :, 0])
