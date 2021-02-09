import torch
import flownet2.models as models

net = models.FlowNetC()
x = torch.randn(5, 6, 128, 128)
o = net(x)
print(o.shape)