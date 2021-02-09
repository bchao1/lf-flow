import torch
import torch.nn as nn
import numpy as np

# spatial-angular resolution

class DisparityNet(nn.Module):
    def __init__(self, views):
        super(DisparityNet, self).__init__()
        self.views = views
        self.conv1 = nn.Sequential(
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
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
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

        # attention map for alpha blending
        self.attn_map = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=self.views, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """ 
            Args: 
                x: input stereo image. Shape (N, 6, H, W)
            Returns:
                horizontal flow: (N, 1, H, W)
        """
        h1 = self.conv1(x)
        disp = self.conv2(h1)
        #attn = self.attn_map(h1)
        return disp
    
class DepthNet(nn.Module):
    """  
    
        Depth Estimation Network as defined in the ICCV '17 paper .
        Input:
            - Single RGB image (N, 3, H, W)
            - Number of views k
        Output:
            Depth map at each view (N, k, H, W)
    """

    def __init__(self, views):
        super(DepthNet, self).__init__()
        self.views = views
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
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
            nn.Conv2d(in_channels=16, out_channels=self.views, kernel_size=3, padding=1),
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
    def __init__(self, in_channels, out_channels, hidden=128):
        super(LFRefineNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
            nn.Conv3d(self.hidden, self.out_channels, (3, 3, 3), padding=1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        #print(x.shape)
        # x: shape: (b, n*n, 3, h, w)
        res = self.net(x)
        out = x[:, :self.out_channels, :, :, :] + res
        out = torch.clamp(out, -1, 1) # important!
        return out


if __name__ == "__main__":
    views = 81
    rnet = LFRefineNet(in_channels=views,out_channels=views).cuda()
    dnet = DisparityNet(views).cuda()
    shufflenet = LFRefineNetV2().cuda()
    x = torch.randn(5, views, 3, 128 , 128).cuda()
    o = shufflenet(x)
    print(o.shape)
    #x = torch.randn(5, in_channels, 4, 4, 4).cuda()
    #o = rnet(x)
    #print(o.shape)
    #x = torch.randn(5, views, 3, 20 , 20).cuda()
    #dnet = DepthNet(views=views).cuda()
    #o = rnet(x)
    #print(o.shape)