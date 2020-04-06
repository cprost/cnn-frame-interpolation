# This is a standalone file for creating datasets and lists of files
# The dataset used is Vimeo90K available here: http://toflow.csail.mit.edu/

import os

IM_WIDTH = 448
IM_HEIGHT = 256

def create_img_list():
    data_folder = "./testdata/"  # must be folder containing sub-folders of images, as per Vimeo90K

    filelist = []

    for path, subdirs, files in os.walk(data_folder):
        for name in files:
            if name == "im1.png":
                filelist.append(path)  # we just need the directory
            #elif name == "im3.png":
                #x_3.append(os.path.join(path, name))
            #elif name == "im2.png":
                #y.append(os.path.join(path, name))

    with open("file_list.txt", "w") as file:
        for s in filelist:
            file.write(str(s) + "\n")

def create_npy():
    x_1 = []
    y = []
    x_3 = []

    sample_list = []

    for path, subdirs, files in os.walk("testdata"):
        for name in files:
            #print(os.path.join(path, name))
            if name == "im1.png":
                x_1.append(os.path.join(path, name))
                sample_list.append(path)
            elif name == "im3.png":
                x_3.append(os.path.join(path, name))
            elif name == "im2.png":
                y.append(os.path.join(path, name))

    # should all be the same
    print(len(x_1))
    print(len(x_3))
    print(len(y))

    #convert from uint8 to 0-1 float before inputting to CNN
    X_data = np.zeros(shape=(samples, IM_HEIGHT, IM_WIDTH, 6), dtype="uint8")
    Y_data = np.zeros(shape=(samples, IM_HEIGHT, IM_WIDTH, 3), dtype="uint8")
    data_out = np.zeros(shape=(samples, IM_HEIGHT, IM_WIDTH, 9), dtype="uint8")

    for i in range(samples):
        data_out[i, :, :, :3] = imageio.imread(x_1[i])
        data_out[i, :, :, 3:6] = imageio.imread(x_3[i])
        data_out[i, :, :, 6:] = imageio.imread(y[i])

    data_out = data_out.astype("float32") / 255
    np.save("./test_data/test_data.npy", data_out)