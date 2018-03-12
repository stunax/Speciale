import config
import h5py
import numpy as np
from scipy.misc import imsave


path = 'labels/'


def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    w, h = 1024, 1024
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


X = []
for i, h5fn in enumerate(config.h5s):
    h5f = h5py.File(h5fn, 'r+')
    X.extend([(h5f, df, h5fn.rsplit("/", 1)[1])
              for df in h5f.keys() if df in config.labels[i]])

for h5f, df, h5fn in X:
    image = np.array(h5f[df]).reshape((1024, 1024))
    image = to_rgb2(image)
    annotation = np.array(h5f["anno" + df])
    imsave(path + h5fn + df + ".png", image)

    image[annotation == 1] = np.array([[0, 255, 0]])
    image[annotation == -1] = np.array([[0, 255, 255]])
    imsave(path + h5fn + "anno" + df + ".png", image)
