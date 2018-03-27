import h5py
import numpy as np
from imageio import imwrite
from sklearn.cluster import KMeans
from model_data import Model_data
import config
import random

h5s = config.get_h5(ignore_1_2=True)

datam = Model_data(
    kernel_size=(9, 9, 1), remove_unlabeled=False, one_hot=False,
    flat_features=True, from_h5=True, bag_size=11)

Xtrain, ytrain = datam.handle_images(h5s)
datam.annotation_groupname = ""
sub_Xtrain = Xtrain[np.random.choice(Xtrain.shape[0], 3000000, False)]
model = KMeans(n_clusters=16, n_jobs=6)
model.fit(sub_Xtrain)


X = []

for h5fn in config.h5s[2:]:
    h5f = h5py.File(h5fn, 'r+')
    for df in h5f.keys():
        if len(df) > 6 or random.uniform(0, 1) < 0.8:
            continue
        image, _ = datam.handle_images([(h5f, df)])
        pred = model.predict(image).astype(np.uint8).reshape(1024, 1024)
        fname = "%s/kmeans_labels/%s/%s_pred.png" % (
            config.data_path, h5fn[-4], df)
        imwrite(fname, pred)
        fname = "%s/kmeans_labels/%s/%s.png" % (
            config.data_path, h5fn[-4], df)
        imwrite(fname, image)
