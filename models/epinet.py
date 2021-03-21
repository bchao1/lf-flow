import torch
import torch.nn as nn
import numpy as np

class EPIVolumeNet(nn.Module):
    def __init__(self, hidden = 128):
        super(EPIVolumeNet, self).__init__()
        self.hidden = 128

        self.net = nn.Sequential(
            nn.Conv3d(3, self.hidden, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace = True),
            nn.Conv3d(self.hidden, self.hidden, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace = True),
            nn.Conv3d(self.hidden, self.hidden, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace = True),
            nn.Conv3d(self.hidden, 3, kernel_size = 3, padding = 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        res = self.net(x)
        x = x + res
        x = torch.clip(x, -1, 1)  # Tanh clipping
        return x

if __name__ == "__main__":
    net = EPIVolumeNet().cuda()
    x = torch.randn(5, 3, 7, 128, 128).cuda()
    o = net(x)
    print(o.shape)