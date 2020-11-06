import numpy as np

if __name__ == '__main__':
    datasets = ['hci', 'inria', 'stanford']
    for d in datasets:
        print(d)
        v = np.load('temp/psnr_v_{}.npy'.format(d))
        h = np.load('temp/psnr_h_{}.npy'.format(d))
        print("v mean: {}".format(np.mean(v)))
        print("h mean: {}".format(np.mean(h)))
    v = np.load('temp/psnr_v_{}.npy'.format('hci'))
    h = np.load('temp/psnr_h_{}.npy'.format('hci'))
    print(np.argmax((h - v)**2))
    print(h[6])
    print(v[6])