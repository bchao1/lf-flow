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
    def __init__(self, views, with_depth=False):
        super(LFRefineNet, self).__init__()
        self.out_channels = views
        if with_depth:
            self.in_channels = views * 2
        else:
            self.in_channels = views
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
            nn.Conv3d(128, self.out_channels, (3, 3, 3), padding=1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        res = self.net(x) # residual correction term to the input coarse light field
        #print(res.shape)
        return x[:, :self.out_channels, :, :, :] + res

if __name__ == "__main__":
    views = 81
    rnet = LFRefineNet(views=views, with_depth=True).cuda()
    #dnet = DisparityNet().cuda()
    #x = torch.randn(5, in_channels, 4, 4, 4).cuda()
    #o = rnet(x)
    #print(o.shape)
    x = torch.randn(5, views, 3, 20 , 20).cuda()
    #dnet = DepthNet(views=views).cuda()
    o = rnet(x)
    print(o.shape)