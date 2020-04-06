import numpy as np
import random

# Make sure to adjust the image dimensions in train_cnn.py too!
BATCH_SIZE = 32  # images per batch, usually at 64
IM_HEIGHT = 256
IM_WIDTH = 448

def gen_batch(file_list, total_samples):
    # NOT USED ON COLAB, LIMITED FILE READ RATE

    #file_list = []
    #with open("/content/drive/My Drive/Colab Notebooks/file_list_short.txt", "r") as filestream:
    #    for line in filestream:
    #        file_list.append(line.strip())

    while True:
        file_subset = random.sample(file_list, BATCH_SIZE)

        X_data = np.zeros(shape=(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, 6), dtype="uint8")
        Y_data = np.zeros(shape=(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, 3), dtype="uint8")

        for i in range(0, BATCH_SIZE):
            X_data[i, :, :, :3] = imageio.imread(os.path.join("/content/drive/My Drive/Colab Notebooks/" + str(file_subset[i]), "im1.png"))
            X_data[i, :, :, 3:] = imageio.imread(os.path.join("/content/drive/My Drive/Colab Notebooks/" + str(file_subset[i]), "im3.png"))
            Y_data[i, :, :, :] = imageio.imread(os.path.join("/content/drive/My Drive/Colab Notebooks/" + str(file_subset[i]), "im2.png"))

        X_data = X_data.astype("float32") / 255
        Y_data = Y_data.astype("float32") / 255

        yield X_data, Y_data

def gen_batch_2(dataset_list):
    dataset_num = random.randint(0, 3)  # adjust upper bound to number of datasets available
    dataset = dataset_list[dataset_num]

    while True:
        dataset_len = dataset.shape[0]
        sample_index = np.random.randint(0, dataset_len, BATCH_SIZE)

        X_data = np.zeros(shape=(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, 6), dtype="uint8")
        Y_data = np.zeros(shape=(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, 3), dtype="uint8")

        for i in range(0, BATCH_SIZE):
            curr_index = sample_index[i]
            X_data[i, :, :, :3] = dataset[curr_index, :, :, :3]
            X_data[i, :, :, 3:] = dataset[curr_index, :, :, 3:6]
            Y_data[i, :, :, :] = dataset[curr_index, :, :, 6:]

            # Data augmentation, random chance for a sample to be modified
            # into a "new" sample, generating a broader dataset

            # flip temporal order i.e. frames are playing backwards
            if random.randint(0, 1):
                X_data[i, :, :, :3] = dataset[curr_index, :, :, 3:6]
                X_data[i, :, :, 3:] = dataset[curr_index, :, :, :3]

            # flip images along horizontal axis
            if random.randint(0, 1):
                X_data[i, :, :, :3] = np.flip(X_data[i, :, :, :3], 1)
                X_data[i, :, :, 3:] = np.flip(X_data[i, :, :, 3:], 1)
                Y_data[i, :, :, :] = np.flip(Y_data[i, :, :, :], 1)

            # flip images along vertical axis
            if random.randint(0, 1):
                X_data[i, :, :, :3] = np.flip(X_data[i, :, :, :3], 0)
                X_data[i, :, :, 3:] = np.flip(X_data[i, :, :, 3:], 0)
                Y_data[i, :, :, :] = np.flip(Y_data[i, :, :, :], 0)

        X_data = X_data.astype("float32") / 255
        Y_data = Y_data.astype("float32") / 255

        yield X_data, Y_data