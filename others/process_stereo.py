import numpy as np 
from PIL import Image

img = np.array(Image.open("../data/bunny_stereo.jpg"))
h, w, c = img.shape
left = img[:, :w//2, :]
right = img[:, w//2:, :]
Image.fromarray(left).save("../data/bunny_left.png")
Image.fromarray(right).save("../data/bunny_right.png")


