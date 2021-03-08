import sys
sys.path.append("..")

import cv2
from PIL import Image
import numpy as np
import numpy.fft as fft

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

from lf_datasets import HCIDataset
import utils
import metrics
#dataset = HCIDataset(root="../../../../mnt/data2/bchao/lf/hci/full_data/dataset.h5", train=False)

def to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def myfft(img):
    return fft.fftshift(fft.fft2(img))

lf = np.load("lf.npy") # dataset.get_single_lf(0)

lf_u, lf_v = lf.shape[0], lf.shape[1] # (u, v) angular resolution
lf_h, lf_w = lf.shape[2], lf.shape[3] # (h, w) spatial resolution

center_v = lf[lf_u // 2, lf_v // 2]
center_vf = np.array(myfft(to_gray(center_v)), dtype=np.complex128)

vfs = np.zeros(lf.shape[:-1], dtype=np.complex128)
dfs = np.zeros(lf.shape[:-1], dtype=np.complex128)

for i in range(lf_u):
    for j in range(lf_v):
        vfs[i, j] = myfft(to_gray(lf[i, j]))
        dfs[i, j] = vfs[i, j] / center_vf

"""
u, v = (0, 0)
view = lf[u, v] # view to reconstruct
df = dfs[u, v]
std = np.std(df)
mean = np.mean(df)
s = 1

mask = np.where(df > mean + s * std)
k = vfs[u, v] / center_vf
k[mask] = 0

recon = np.abs(fft.ifft2(fft.ifftshift(center_vf * k)))

score = metrics.psnr(recon, to_gray(view))
print(score)

#utils.animate_sequence(20*np.log(dfs[6, :]).astype(np.uint8))
plt.imshow(20*np.log(np.abs(center_vf)), cmap='gray')
plt.show()
#plt.imshow(np.clip(20 * np.log(np.abs(k)), 0, 255).astype(np.uint8), cmap='gray')
#plt.show()
#utils.animate_sequence(dfs[4])
"""

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(lf_w)
y = np.arange(lf_h)
X, Y = np.meshgrid(x, y)
Z = df_mag

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""