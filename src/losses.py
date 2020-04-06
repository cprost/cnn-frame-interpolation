# This defines custom loss functions for training the CNN
# These have better empirical performance against MSE

import keras

def ssim(img_true, img_interp):
    # taking 1 - SSIM since we want to minimize the SSIM loss, as SSIM == 1 means the images are identical
    return keras.backend.mean(1 - tf.image.ssim(img_true, img_interp, max_val=1.0))

def charbonnier(img_true, img_interp):
    epsilon = 0.00001
    return keras.backend.sqrt(keras.backend.square(img_true - img_interp) + epsilon * epsilon)