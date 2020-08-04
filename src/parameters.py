'''
This file contains operational parameters and hyperparameters
'''

BATCH_SIZE = 4  # depends on GPU usage
SHUFFLE_BUF = 1000
EPOCHS = 10
IMG_HEIGHT = 360  # 360
IMG_WIDTH = 480  # 480
ADAM_LR = 0.0001  # can reduce on later epochs with a callback, if necessary

LAMBDA_R = 0.8 * 255
LAMBDA_P = 0.005
LAMBDA_W = 0.4 * 255
LAMBDA_S = 1.0
