import torch
import torch.nn as nn



class Network(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Network, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=100, kernel_size=7, padding=3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=5, padding=2),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=self.out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x).squeeze()
    
if __name__ == '__main__':
    x = torch.randn(5, 200, 128, 128)
    net = DisparityNet()
    o = net(x)
    print(o.shape)