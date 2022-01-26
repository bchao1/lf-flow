from PIL import Image

path = "../temp/mymodel/hci/20211226_1420_final/10000/lf_3.gif"
path = "../temp/2crop_wide/hci/20211228_2212_wide/6000/lf3.gif"
im = Image.open(path)
center_frame = im.n_frames // 2
im.seek(center_frame) 
im.save("center2.png")