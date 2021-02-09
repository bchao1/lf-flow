import sys
sys.path.append("..")
import torch
#import models.flownet2.models as flownet_models
import models.lf_net as lfnet_models
import models.net_siggraph as siggraph16_models
import models.unet as unet_models
#import models.PSMNet.models as psm_models

from utils import count_parameters

#basic = psm_models.basic(4)
#stackhourglass = psm_models.stackhourglas(4)

class Args():
    fp16 = False
    rgb_max = 255

args = Args()
views = 9*9
#flownet = flownet_models.FlowNetC.FlowNetC(args)
disparity_net_ours = lfnet_models.DisparityNet(views)
refine_net_ours = lfnet_models.LFRefineNetV2(views, views, views)

depth_net_iccv17  = lfnet_models.DepthNet(views)
refine_net_iccv17 = lfnet_models.LFRefineNet(views*2, views, 128)

refine_net_sig16 = siggraph16_models.Network(in_channels=3*4+1, out_channels=3)
depth_net_sig16 = siggraph16_models.Network(in_channels=100*2, out_channels=1)

unet = unet_models.UNet(n_channels=3, n_classes=1, bilinear=True)

print("Unet:", count_parameters(unet) / 10**6)
#print("PSMNet (basic)", count_parameters(basic) / 10**6)
#print("PSMNet (stacked hour glass)", count_parameters(stackhourglass) / 10**6)
print("")
print("Depth nets:")
print("Ours: ", count_parameters(disparity_net_ours)/ 10**6) # here is not right! no attention. less than iccv17
print("ICCV '17: ", count_parameters(depth_net_iccv17)/ 10**6)
print("SIGGRAPH '16: ", count_parameters(depth_net_sig16)/ 10**6)
print("")
print("Refine nets:")
print("Ours: ", count_parameters(refine_net_ours)/ 10**6)
print("ICCV '17: ", count_parameters(refine_net_iccv17)/ 10**6)
print("SIGGRAPH '16: ", count_parameters(refine_net_sig16)/ 10**6)
print("")
print("Total")
print("Ours: ", count_parameters(disparity_net_ours)/ 10**6+count_parameters(refine_net_ours)/ 10**6)
print("ICCV '17: ", count_parameters(depth_net_iccv17)/ 10**6+count_parameters(refine_net_iccv17)/ 10**6)
print("SIGGRAPH '16: ", count_parameters(depth_net_sig16)/ 10**6+count_parameters(refine_net_sig16)/ 10**6)