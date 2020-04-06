import numpy as np
from skimage.measure import compare_ssim, compare_psnr

# load the MSE model, since saving a custom loss function model doesn't save the loss function with it.
# the weights for a custom loss model can be loaded after, as load_model is just for the network structure
restored_model = tf.keras.models.load_model("./models/cnn_mse_100epoch.h5")

# MUST RESTORE WEIGHTS IF USING CUSTOM LOSS e.g. CHARBONNIER OR SSIM
restored_model.load_weights("./weights/weights_ssim_100ep.hdf5")
restored_model.summary()

def gen_images():
    img_data = np.load("./dataset/data_test.npy")
    img_count = img_data.shape[0]

    img_input = img_data[:, :, :, :6]
    img_true = img_data[:, :, :, 6:]
    print("Input shape:", img_input.shape)  # dim check

    img_pred = restored_model.predict(img_input)
    print("Output shape:", img_pred.shape)

    for i in range(0, img_count):
        imageio.imwrite("./results/" + str(i) + "_true.jpg", img_data[i, :, :, 6:])
        imageio.imwrite("./results/" + str(i) + "_pred.jpg", img_pred[i, :, :, :])

def evaluate():
    ssim_list = np.zeros(img_count)
    psnr_list = np.zeros(img_count)

    for i in range(0, img_count):
        if i % 200 == 0:
            print("Completed", i, "images")
        ssim_list[i] = compare_ssim(img_true[i], img_pred[i], data_range=img_pred[i].max() - img_pred[i].min(), multichannel=True)
        psnr_list[i] = compare_psnr(img_true[i], img_pred[i], data_range=img_pred[i].max() - img_pred[i].min())

    print("The average SSIM for the model is:", np.mean(ssim_list))
    print("The SSIM standard deviation for the model is:", np.std(ssim_list))

    print("\nThe average PSNR for the model is:", np.mean(psnr_list))