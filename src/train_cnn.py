import os
import random
import numpy as np
import tensorflow as tf
import keras

from callbacks import cb_checkpoint, cb_reduce_lr
from losses import ssim, charbonnier
from generator import gen_batch_2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 448

def create_network(input_shape):
    # I used https://github.com/zhixuhao/unet/blob/master/model.py as a template for tensorflow/keras

    cnn_input = keras.layers.Input(input_shape)

    # downsampling section of CNN

    conv2d_1_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    conv2d_1_2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2d_1_1)
    maxpool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2d_1_2)

    conv2d_2_1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool_1)
    conv2d_2_2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_2_1)
    maxpool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2d_2_2)

    conv2d_3_1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(maxpool_2)
    conv2d_3_2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_3_1)
    maxpool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2d_3_2)

    conv2d_4_1 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(maxpool_3)
    conv2d_4_2 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_4_1)
    maxpool_4 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2d_4_2)

    conv2d_5_1 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(maxpool_4)
    conv2d_5_2 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2d_5_1)

    # upsampling section of CNN

    upsample_6 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv2d_5_2), conv2d_4_2])  # merge two prior outputs together
    conv2d_6_u = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(upsample_6)
    conv2d_6_u = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_6_u)

    upsample_7 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv2d_6_u), conv2d_3_2])
    conv2d_7_u = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(upsample_7)
    conv2d_7_u = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_7_u)

    upsample_8 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv2d_7_u), conv2d_2_2])
    conv2d_8_u = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(upsample_8)
    conv2d_8_u = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_8_u)

    upsample_9 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(conv2d_8_u), conv2d_1_2])
    conv2d_9_u = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upsample_9)
    conv2d_9_u = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2d_9_u)

    # maps to 3 channel (RGB) output of same size as each input image
    cnn_output = keras.layers.Conv2D(3, (1, 1), activation='relu', padding='same')(conv2d_9_u)

    model = keras.models.Model(input=cnn_input, output=cnn_output)
    model.summary()  # prints output of merged layers

    return model


# uncomment each of these as they are uploaded, each is 3GB
# data_1 = np.load("./dataset/data_1.npy")
# data_2 = np.load("./dataset/data_2.npy")
# data_3 = np.load("./dataset/data_3.npy")
# data_4 = np.load("./dataset/data_4.npy")
# dataset_list = [data_1, data_2, data_3, data_4]

total_samples = 0
for ds in dataset_list:
    total_samples = total_samples + ds.shape[0]

input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 6)  # (256, 448, 3), change to 6 channels since appending

print("Input shape:", input_shape)
print("Total samples:", total_samples)

# may add others depending on how the model converges
callback_list = [cb_checkpoint, cb_reduce_lr]

loss_func = charbonnier  # change to whichever gives best empirical performance

model = create_network(input_shape)  # W=448 H=256 CHANNELS=6 (first 3 for im1, last 3 for im2)
optim = keras.optimizers.Adam(lr=0.001)

model.compile(loss=loss_func, optimizer=optim)

testtrain = model.fit_generator(
    generator=gen_batch_2(dataset_list),
    steps_per_epoch=(total_samples // BATCH_SIZE),
    callbacks=callback_list,
    epochs=100
)

model.save("./models/unet_mse.h5")  # outputs model structure and weights from training
