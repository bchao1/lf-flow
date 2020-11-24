import torch
import torch.nn as nn

class DisparityNet(nn.Module):
    def __init__(self):
        super(DisparityNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        """ 
            Args: 
                x: input stereo image. Shape (N, 6, H, W)
            Returns:
                horizontal flow: (N, 1, H, W)
        """
        return self.net(x)
    
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
    in_channels = 9
    rnet = LFRefineNet(in_channels=in_channels).cuda()
    dnet = DisparityNet().cuda()
    x = torch.randn(5, in_channels, 4, 4, 4).cuda()
    o = rnet(x)
    print(o.shape)