import os
import random
import numpy as np
import tensorflow as tf
import keras
import imageio

# load the MSE model, since saving a custom loss function model doesn't save the loss function with it.
# the weights for a custom loss model can be loaded after, as load_model is just for the network structure
restored_model = tf.keras.models.load_model("./models/cnn_mse_100epoch.h5")

# MUST RESTORE WEIGHTS IF USING CUSTOM LOSS e.g. CHARBONNIER OR SSIM
restored_model.load_weights("./weights/weights_ssim_100ep.hdf5")
restored_model.summary()

img_data = np.load("./demo_data/demo_data.npy")
img_count = img_data.shape[0]

img_input = img_data[:, :, :, :6]
print("Input shape:", img_input.shape)  # dim check

img_pred = restored_model.predict(img_input)
print("Output shape:", img_pred.shape)

for i in range(0, img_count):
    imageio.imwrite("./results/" + str(i) + "_true.jpg", img_data[i, :, :, 6:])
    imageio.imwrite("./results/" + str(i) + "_pred.jpg", img_pred[i, :, :, :])