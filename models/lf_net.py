import torch
import torch.nn as nn
import numpy as np

# spatial-angular resolution

class DisparityNet(nn.Module):
    def __init__(self, views, attention=True):
        super(DisparityNet, self).__init__()
        self.views = views
        self.attention = attention

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
        if self.attention:
            self.attn_map = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=self.views, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.attn_map = nn.Identity()
    
    def forward(self, x):
        """ 
            Args: 
                x: input stereo image. Shape (N, 6, H, W)
            Returns:
                horizontal flow: (N, 1, H, W)
        """
        h1 = self.conv1(x)
        disp = self.conv2(h1)
        attn = self.attn_map(h1)
        return disp, attn
    
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
    def __init__(self, in_channels, out_channels, hidden=128, residual=True):
        super(LFRefineNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.residual = residual

        # spatio
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )

        # angular
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(self.hidden, self.out_channels, (3, 3, 3), padding=1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        #print(x.shape)
        # x: shape: (b, n*n, 3, h, w)
        b, views, c, h, w = x.shape
        lf_res = int(np.sqrt(views))
        # residual correction term to the input coarse light field
        h1 = self.conv1(x) 
        #print("after conv1", res.shape)
        ## shuffle 
        #res = res.view(b, lf_res, lf_res, c, h, w) # 0, 1, 2, 3, 4, 5
        #res = res.permute(0, 4, 5, 3, 1, 2)
        #res = res.view(b, h*w, c, lf_res, lf_res)
        #print("after shuffling", res.shape)
        h2 = self.conv2(h1)
        #print(res.shape)
        h3 = self.conv3(h2)
        #print(res.shape)
        h4 = self.conv4(h3)
        #print(res.shape)
        res = self.conv5(h4) # output refinement
        #print(res.shape)

        #print(res.shape)
        if self.residual:
            out = x[:, :self.out_channels, :, :, :] + res
        else:
            out = res # end2end refinement
        out = torch.clamp(out, -1, 1) # important!
        return out

class LFRefineNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=128, residual=True):
        super(LFRefineNetV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.residual = residual

        # spatio
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )

        # angular
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.hidden, self.hidden, (3, 3, 3), padding=1),
            nn.BatchNorm3d(self.hidden),
            nn.ELU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(self.hidden, self.out_channels, (3, 3, 3), padding=1),
            nn.Tanh() # [-1, 1]
        )


    def spatio_angular_shuffle(self, x):
        b, views, c, h, w = x.shape
        lf_res = int(np.sqrt(views))
        x = x.contiguous().view(b, lf_res, lf_res, c, lf_res, h // lf_res, lf_res, w // lf_res) 
        # (b, n, n, c, n, h/n, n, w/n), 0, [1, 2] ,3, [4] ,5 ,[6] ,7
        x = x.contiguous().permute(0, 4, 6, 3, 1, 5, 2, 7)
        x = x.contiguous().view(b, views, c, h, w)
        return x

    def forward(self, x):
        # spatio conv
        res = self.conv1(x) # (b, n*n, 128, h, w)
        # to angular
        res = self.spatio_angular_shuffle(res)

        # angular conv
        res = self.conv2(res)
        # to spatio
        res = self.spatio_angular_shuffle(res)

        # spatio conv
        res = self.conv3(res)
        # to angular
        res = self.spatio_angular_shuffle(res)

        # angular conv
        res = self.conv4(res)
        # to spatio
        res = self.spatio_angular_shuffle(res)

        # spatio conv
        res = self.conv5(res)

        if self.residual:
            out = x + res
        else:
            out = res # end2end refinement
        out = torch.clamp(out, -1, 1) # important!
        return out



if __name__ == "__main__":
    views = 49
    x = torch.randn(5, views, 3, 126, 126).cuda()
    net = LFRefineNetV2(views, views, views).cuda()
    o = net(x)
    print(o.shape) 