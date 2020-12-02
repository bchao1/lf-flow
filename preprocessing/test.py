import os 
from PIL import Image
import numpy as np

from shutil import copyfile

folder = "../../../../mnt/data2/bchao/lf/stanford_lytro/occlusions/raw"
files = os.listdir(folder)

#img = Image.open(os.path.join(folder, files[0]))
#img = np.array(img)
#print(img.shape)
copyfile(os.path.join(folder, files[0]), "test.png")