import skimage.metrics as metrics

def psnr(img, target):
    return metrics.peak_signal_noise_ratio(target, img)

def mse(img, target):
    return metrics.mean_squared_error(img, target)

