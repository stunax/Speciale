import h5py
import numpy as np
from imageio import imwrite
from sklearn.cluster import KMeans
from model_data import Model_data
from helpers import cat_image_to_gray
import os.path
import config
import random

h5s = config.get_h5(ignore_1_2=True)

datam = Model_data(
    kernel_size=(9, 9, 1), remove_unlabeled=False, one_hot=False,
    flat_features=True, from_h5=True, bag_size=1, annotation_groupname="")
datam2 = Model_data(
    kernel_size=(1, 1, 1), remove_unlabeled=False, one_hot=False,
    flat_features=True, from_h5=True, bag_size=1, annotation_groupname="")


h5s = {str(fn): h5py.File(config.data_path + str(fn) + ".h5", "a")
       for fn in range(1, 6)}

datam.bag_size = 1
X = []
model = KMeans(n_clusters=16, n_jobs=30,
               precompute_distances=True)


def to_rgb1a(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def func(x):
    h5f, df = x
    if len(df) > 6 or random.uniform(0, 1) < 0.95:
        return False
    fname = "%s%s/%s_pred.png" % (
        config.kmeans_path, h5f.filename[-4], df)
    if os.path.isfile(fname):
        return True
    print(x)
    image, _ = datam.handle_images([(h5f, df)])
    if np.sum(image) == 0:
        print("Empty")
        return False
    model.fit(image)
    pred = model.predict(image)
    uniques = np.unique(pred, return_counts=True)
    if len(uniques[0]) < 16:
        return False
    model.init = model.cluster_centers_
    pred = pred.reshape(1024, 1024)
    imwrite(fname, pred)
    image, _ = datam2.handle_images([(h5f, df)])
    image = image.reshape(1024, 1024)
    image = to_rgb1a(image.astype(np.uint8))
    fname = "%s/kmeans_labels/%s/%s.png" % (
        config.data_path, h5f.filename[-4], df)
    imwrite(fname, image)
    return True


for h5fn in config.h5s[2:]:
    h5f = h5py.File(h5fn, 'r+')
    [func((h5f, df)) for df in h5f.keys()]
