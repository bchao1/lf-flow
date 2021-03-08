import sys
sys.path.append("..")

import cv2
from PIL import Image
import numpy as np

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

import utils

lf = np.load("lf.npy") # dataset.get_single_lf(0)

def to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

lf = lf.reshape(lf.shape[0] * lf.shape[1], *lf.shape[2:])
lum = np.zeros(lf.shape[:-1]) # only one channel for luminance
for i, view in enumerate(lf):
    lum[i] = to_gray(view)

lf_mean = np.mean(lum, axis=0).astype(np.uint8)
lf_var = np.var(lum, axis=0).astype(np.uint8)

Image.fromarray(lf_mean).save("mean.png")
Image.fromarray(lf_var).save("var.png")


