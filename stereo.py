import numpy as np

import pyflow
import utils
import metrics

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

def test_stereo2lf():
    from lf_datasets import HCIDataset
    from PIL import Image

    dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5")
    lf = np.array(dataset.get_single_lf(1), dtype=np.float) * 1.0 / 255
    lf_res = lf.shape[0]
    left_img = lf[lf_res // 2, 1]
    right_img = lf[lf_res // 2, lf_res - 2]
    
    d = stereo2lf(right_img, left_img)
    lf_syn = utils.generate_lf(left_img, d * 1.0 / (lf_res - 3), 3)
    lf_target = lf[lf_res // 2 - 1: lf_res // 2 + 2, 0: 3]

    mse = metrics.mse(lf_syn, lf_target)
    psnr = metrics.psnr(lf_syn, lf_target)
    print("MSE: {}".format(mse))
    print("PSNR: {}".format(psnr))
    return
    
if __name__ == '__main__':
    test_stereo2lf()