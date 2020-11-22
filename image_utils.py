from PIL import Image 
import numpy as np
import time
import cv2

import torch 
import torchvision

def save_image(img, path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    
def sample_stereo_index(lf_size):
    """ Get the lf coordinate of stereo subimages """
    mid_index = lf_size // 2
    row_index = np.random.randint(1, lf_size - 1)
    left_stereo_index = np.random.randint(1, mid_index)
    right_stereo_index = np.random.randint(mid_index + 1, lf_size - 1)
    return row_index, left_stereo_index, right_stereo_index

def sample_lf_index(lf_size):
    """ Get a lf coordinate of a lf subimage """
    row_index = np.random.randint(lf_size)
    col_index = np.random.randint(lf_size)
    return row_index, col_index

def crop_image(img):
    """ Input 2D light image (H, W, C) """
    pass

def resize_lf(lf, sz):
    """ 
        Args: 
            lf: 4D light field (L, L, H, W, C) 
            sz: downsample size
        Returns: 
            Downsampled light field (L, L, sz, sz, C)
    """
    lfh, lfw, h, w, c = lf.shape
    new_lf = np.zeros((lfh, lfw, sz, sz, c))
    for i in range(lfh):
        for j in range(lfw):
            view = lf[i, j, :, :, :]
            new_lf[i, j, :, :, :] = cv2.resize(view, (sz, sz))
    return new_lf

def crop_lf(lf, sz):
    """ 
        Args: 
            lf: 4D light field (L, L, H, W, C) 
            sz: crop size
        Returns: 
            Cropped light field (L, L, sz, sz, C)
    """
    _, _, h, w, _ = lf.shape
    assert sz <= h and sz <= w

    top_left_i = np.random.randint(low = 0, high = h - sz + 1) # [0, h - sz]
    top_left_j = np.random.randint(low = 0, high = w - sz + 1) # [0, w - sz]
    cropped_lf = lf[:, :, top_left_i: top_left_i + sz, top_left_j: top_left_j + sz, :]
    return cropped_lf

def get_lf_horizonal_view(lf):
    """ Input 4D light field (L, L, H, W, C) """

    lf_res = lf.shape[0] # light field angular resolution. row view [0 ~ lf_res - 1]
    row_idx = np.random.randint(low = 1, high=lf_res - 1)
    
    d = np.random.randint(low = 1, high = lf_res // 2)
    left_col_idx, right_col_idx = d, lf_res - d - 1
    return lf[row_idx, left_col_idx], lf[row_idx, right_col_idx], 

def get_lf_vertical_view(lf):
    pass

def test_im_utils():
    from lf_datasets import HCIDataset, StanfordDataset, INRIADataset
    
    from PIL import Image

    dataset = HCIDataset(root="../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", use_all=True)
    lf = dataset.get_single_lf(0)
    print(lf.shape)
    #stereo_pair = get_lf_horizonal_view(lf)

    # Test cropping
    start = time.time()
    cropped_lf = crop_lf(lf, 256)
    print("Original light field shape: {}".format(lf.shape))
    print("Cropped light field shape: {}".format(cropped_lf.shape))
    print("Time spent: {}\n".format(time.time() - start))

    # Test multiview <-> lf
    start = time.time()
    multiview_img = lf_to_multiview(lf)
    print("Multiview image array shape: {}".format(multiview_img.shape))
    #Image.fromarray(multiview_img.astype(np.uint8)).save('temp/multiview.png')
    #exit()
    converted_back_lf = multiview_to_lf(multiview_img, 512, 512)
    print("Asserting lf conversion ...")
    assert np.array_equal(lf, converted_back_lf)
    print("Time spent: {}\n".format(time.time() - start))

    # Test light field rotation
    start = time.time()
    rot_lf = rotate_lf(lf)
    print("Asserting lf rotation ...")
    assert rot_lf.shape == lf.shape
    rotback_lf = rotate_lf(rotate_lf(rotate_lf(rotate_lf(lf)))) # rotate 360 = original
    assert np.array_equal(rotback_lf, lf)
    print("Time spent: {}\n".format(time.time() - start))


def lf_to_multiview(lf):
    """ 
        Args: 
            lf: 4D light field (L, L, H, W, C) 
        Returns
            A 2D (L * H, L * W, C) multiview image array.
    """
    lh, lw, h, w, c = lf.shape
    lf = np.swapaxes(lf, 1, 2)  # (L, H, L, W, C) make into continguous to reshape
    multiview_img = lf.reshape(lh * h, lw * w, c)
    return multiview_img

def multiview_to_lf(mv_img, sz_h, sz_w):
    """ 
        Args: 
            mv_img: 2D (L * sz, L * sz, C) multiview image array.
        Returns
            A 4D (L, L, sz, sz, C) light field.
    """
    h, w, c = mv_img.shape

    assert (h % sz_h == 0) and (w % sz_w == 0)

    #lf = np.empty((h // sz_h, w // sz_w, sz_h, sz_w, c))
    #for i in range(h // sz_h):
    #    for j in range(w // sz_w):
    #        view = mv_img[i*sz_h : (i+1)*sz_h, j*sz_w : (j+1)*sz_w, :]
    #        lf[i, j] = view
    lf = mv_img.reshape(h // sz_h, sz_h, w // sz_w, sz_w, c) # (L, sz, L, sz, C)
    lf = np.swapaxes(lf, 1, 2) # (L, L, sz, sz, C)
    return lf

def rotate_lf(lf):
    """
        Rotate the whole light field by 90 degrees.
    """
    _, _, h, w, c = lf.shape
    mv = lf_to_multiview(lf)
    rot_mv = np.rot90(mv)
    #Image.fromarray(rot_mv.astype(np.uint8)).save('temp/rot_mv.png')
    rot_lf = multiview_to_lf(rot_mv, h, w)
    return rot_lf

if __name__ == '__main__':
    #test_im_utils()
    import utils
    lf = np.load("./experiments/syn_lf_100.npy")
    lf = utils.denorm_tanh(lf)
    #lf = lf.reshape(9, 9, *lf.shape[1:])
    view = lf[0, 0]
    view = torch.tensor(view).unsqueeze(0)
    torchvision.utils.save_image(view, 'test.png')
    #lf = lf.astype(np.uint8)
    #lf = np.swapaxes(lf, 2, 4)
    #lf = np.swapaxes(lf, 2, 3)
    #img = lf_to_multiview(lf)
    #print(img.shape)
    #Image.fromarray(img).save('test.png')
    #lf = np.zeros((9, 9, 128, 128, 3))
    #lf = resize_lf(lf, 64)
    #print(lf.shape)