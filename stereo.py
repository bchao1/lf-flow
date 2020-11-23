import numpy as np

import pyflow
import utils
import image_utils
import metrics
import torch

# HCI (512*512) max disparity: 37

def optical_flow_cfg():
    cfg = {
        "alpha"              : 0.012,
        "ratio"              : 0.75,
        "minWidth"           : 20,
        "nOuterFPIterations" : 7,
        "nInnerFPIterations" : 1,
        "nSORIterations"     : 30,
        "colType"            : 0
    }
    return cfg

def stereo2lf(im1, im2, mode="horizontal"):
    # Return "how much" im1 needs to shift to get im2.
    assert mode in ["horizontal", "vertical", "all"]
    cfg = optical_flow_cfg()
    dx, dy, _ = pyflow.coarse2fine_flow(
        im1.astype(np.float), im2.astype(np.float), 
        cfg["alpha"], 
        cfg["ratio"], 
        cfg["minWidth"], 
        cfg["nOuterFPIterations"], 
        cfg["nInnerFPIterations"],
        cfg["nSORIterations"], 
        cfg["colType"]
    )

    if mode == "horizontal":
        return dx
    elif mode == "vertical":
        return dy
    else:
        return dx, dy

def test_single_stereo():
    from lf_datasets import HCIDataset
    from utils import warp_image, warp_image_batch, normalize
    from image_utils import save_image
    dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", use_all=True)
    lf = np.array(dataset.get_single_lf(0), dtype=np.float) * 1.0 / 255

    lf_res = lf.shape[0]
    left_img = lf[lf_res // 2, 1]
    right_img = lf[lf_res // 2, lf_res - 2]
    target_view = lf[0, lf_res // 2] # top-middle view

    d = stereo2lf(right_img, left_img) / (lf_res - 3)
    
    dx = d * (lf_res // 2 - 1)
    dy = d * (0 - lf_res // 2)
    syn_from_left = warp_image(left_img, dx, dy)
    
    d = -stereo2lf(left_img, right_img) / (lf_res - 3)
    dx = d * (lf_res // 2 - (lf_res - 2))
    dy = d * (0 - lf_res // 2)
    syn_from_right = warp_image(right_img, dx, dy)

    err_left = np.mean((target_view - syn_from_left)**2, axis=-1)
    err_right = np.mean((target_view - syn_from_right)**2, axis=-1)
    
    err_left *= 255
    err_right *= 255
    target_view *= 255 
    syn_from_left *= 255
    syn_from_right *= 255
    save_image(target_view, "./temp/target.png")
    save_image(syn_from_left, "./temp/syn_left.png")
    save_image(syn_from_right, "./temp/syn_right.png")
    save_image(err_left, "./temp/err_left.png")
    save_image(err_right, "./temp/err_right.png")

def test_stereo2lf():
    from lf_datasets import HCIDataset, StanfordDataset, INRIADataset
    from PIL import Image

    dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", use_all=True)
    #dataset = StanfordDataset(root="../../../mnt/data2/bchao/lf/stanford/dataset.h5")
    #dataset = INRIADataset(root="../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5", use_all=True)

    psnr_h_log = []
    psnr_v_log = []
    max_disp = 0
    for i in range(dataset.num_lfs):
        #lf = dataset.get_single_lf(i)
        lf = np.array(dataset.get_single_lf(i), dtype=np.float) * 1.0 / 255
        lf_res = lf.shape[0]
        left_img = lf[lf_res // 2, 1]
        right_img = lf[lf_res // 2, lf_res - 2]
        
        d = stereo2lf(right_img, left_img)
        max_disp = max(max_disp, np.max(np.abs(d.ravel())))
        disp = (utils.normalize(d) * 255).astype(np.uint8)
        #Image.fromarray(disp).save('temp/disp.png')

        d = d * 1.0 / (lf_res - 3) # adjacent-view warp amount, divide by # in-between views
        lf_syn = utils.generate_lf(left_img, d, 3)
        #np.save('temp/lf_syn.npy', lf_syn)
        lf_target = lf[lf_res // 2 - 1: lf_res // 2 + 2, 0: 3]
        
        pixel_shift = 0.5
        
        refocus_syn = utils.refocus(lf_syn, pixel_shift)
        refocus_target = utils.refocus(lf_target, pixel_shift)

        mse = metrics.mse(lf_syn, lf_target)
        psnr_view = metrics.psnr(lf_syn, lf_target)
        psnr_refocus = metrics.psnr(refocus_syn, refocus_target)
        #print("MSE: {}".format(mse))
        print("PSNR (view): {}".format(psnr_view))
        print("PSNR (refocus): {}".format(psnr_refocus))

        horizontal_syn = utils.warp_strip(left_img, d, -1, lf_res - 1, 'horizontal')
        horizontal_target = lf[lf_res // 2, :]
        #print(horizontal_syn.shape, horizontal_target.shape)
        assert horizontal_syn.shape == horizontal_target.shape

        vertical_syn = utils.warp_strip(left_img, d, -(lf_res // 2), lf_res // 2 + 1, 'vertical')
        vertical_target = lf[:, 1]
        #print(vertical_syn.shape, vertical_target.shape)
        assert vertical_syn.shape == vertical_target.shape

        psnr_horizontal = metrics.psnr(horizontal_syn, horizontal_target)
        psnr_vertical = metrics.psnr(vertical_syn, vertical_target)
        print("PSNR (horizontal warp): {}".format(psnr_horizontal))
        print("PSNR (vertical warp): {}".format(psnr_vertical))

        psnr_h_log.append(psnr_horizontal)
        psnr_v_log.append(psnr_vertical)
    print("Max disparity: ", max_disp)
    #np.save('temp/psnr_h_inria.npy', np.array(psnr_h_log))
    #np.save('temp/psnr_v_inria.npy', np.array(psnr_v_log))
    #print("Horizontal PSNR: {}".format(np.mean(psnr_h_log)))
    #print("Vertical PSNR: {}".format(np.mean(psnr_v_log)))
    return

def test_stereo2lf_gradient():
    from lf_datasets import HCIDataset, StanfordDataset, INRIADataset
    
    from PIL import Image
    from skimage import filters

    dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", use_all=True)
    #dataset = StanfordDataset(root="../../../mnt/data2/bchao/lf/stanford/dataset.h5")
    #dataset = INRIADataset(root="../../../mnt/data2/bchao/lf/inria/Dataset_Lytro1G/dataset.h5", use_all=True)


    lf = np.array(dataset.get_single_lf(6), dtype=np.float) * 1.0 / 255
    lf_res = lf.shape[0]
    left_img = lf[lf_res // 2, 1]
    right_img = lf[lf_res // 2, lf_res - 2]
    
    d = stereo2lf(right_img, left_img)
    disp = (utils.normalize(d) * 255).astype(np.uint8)
    #Image.fromarray(disp).save('temp/disp.png')
    d = d * 1.0 / (lf_res - 3) # adjacent-view warp amount, divide by # in-between views
    horizontal_syn = utils.warp_strip(left_img, d, -1, lf_res - 1, 'horizontal')
    horizontal_target = lf[lf_res // 2, :]
    #print(horizontal_syn.shape, horizontal_target.shape)
    assert horizontal_syn.shape == horizontal_target.shape
    vertical_syn = utils.warp_strip(left_img, d, -(lf_res // 2), lf_res // 2 + 1, 'vertical')
    vertical_target = lf[:, 1]
    #print(vertical_syn.shape, vertical_target.shape)

    assert vertical_syn.shape == vertical_target.shape

    horizontal_syn_edge    = filters.sobel(utils.to_gray(horizontal_syn[-1]))
    horizontal_target_edge = filters.sobel(utils.to_gray(horizontal_target[-1]))
    vertical_syn_edge      = filters.sobel(utils.to_gray(vertical_syn[-1]))
    vertical_target_edge   = filters.sobel(utils.to_gray(vertical_target[-1]))

    Image.fromarray((horizontal_syn_edge * 255).astype(np.uint8)).save('temp/hse.png')
    Image.fromarray((horizontal_target_edge * 255).astype(np.uint8)).save('temp/hte.png')
    Image.fromarray((vertical_syn_edge * 255).astype(np.uint8)).save('temp/vse.png')
    Image.fromarray((vertical_target_edge * 255).astype(np.uint8)).save('temp/vte.png')


    print("H edge PSNR: {}".format(metrics.psnr(horizontal_syn_edge, horizontal_target_edge)))
    print("V edge PSNR: {}".format(metrics.psnr(vertical_syn_edge, vertical_target_edge)))

    return
    
if __name__ == '__main__':
    test_stereo2lf()
    #test_stereo2lf_gradient()
    #test_single_stereo()