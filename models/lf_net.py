import torch
import torch.nn as nn

class LFRefineNet(nn.Module):
    def __init__(self, in_channels):
        super(LFRefineNet, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.Conv3d(128, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.Conv3d(128, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.Conv3d(128, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.Conv3d(128, self.in_channels, (3, 3, 3), padding=1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        res = self.net(x) # residual correction term to the input coarse light field
        #print(res.shape)
        return x + res

if __name__ == "__main__":
    net = LFRefineNet(in_channels=81)
    x = torch.randn(8, 81, 3, 128, 128)
    o = net(x)
    print(o.shape)