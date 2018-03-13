import config
import h5py
import model_data
import numpy as np
import matplotlib.pyplot as plt


def report(bag_size, kernel_size, median_filter):
    print("Bagsize:", bag_size)
    print("kernel_size:", kernel_size)
    print("median_filter:", median_filter)


h5f = config.h5s[0]
h5f = h5py.File(h5f, 'r+')
# print(list(h5f.keys()))
s = "083_"
for i in range(31, 40):
    h5f = config.h5s[0]
    h5f = h5py.File(h5f, 'r+')
    train = [(h5f, s + str(i).zfill(2))]

    datam = model_data.Model_data(
        kernel_size=(1, 1, 1), from_h5=True, debug=True,
        bag_size=1, median_time=0, preprocess=True)
    datam.remove_unlabeled = False
    X, _ = datam.handle_images(train)
    X.shape = config.image_size
    fn = config.pred_path + str(i).zfill(2) + "test.png"

    plt.imsave(fn, X, cmap=plt.cm.gray)

    if (np.sum(X) > 1):
        print(X)

    # datam = model_data.Model_data(
    #     kernel_size=(1, 1, 1), from_h5=True, debug=True,
    #     bag_size=1, median_time=0)
    # datam.remove_unlabeled = False
    # X, _ = datam.handle_images(train)
    # X.shape = config.image_size

    # print(X)
