import torch
import torch.nn as nn
import numpy as np

class BaseNetwork(nn.Module):
    """ Network to output N*N masks given single image input """

    def __init__(self, views):
        super(BaseNetwork, self).__init__()

        self.views = views
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.views, kernel_size = 3, padding = 1)
        )
    
    def forward(self, x):
        return self.net(x)

"""
    Questions:
        - Complex representation: cartesian or polar?
"""

class FrequencyMaskingNet(nn.Module):
    """
        Input single image, output N*N frequency masks.
    """

    def __init__(self, views):
        super(FrequencyMaskingNet, self).__init__()

        self.n1 = BaseNetwork(views = views)
        self.n2 = BaseNetwork(views = views)
        self.type = type

    
    def forward(self, x):

        h1 = self.n1(x)
        h2 = self.n2(x)

        return x

if __name__ == '__main__':
    center_view = torch.randn(5, 3, 128, 128)
    net = BaseNetwork(views = 9)

    mask = net(center_view)
    exponent = torch.exp(-2j * np.pi * mask) # cannot use exp, no autograd on complex
    #print(exponent)
        