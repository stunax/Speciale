import h5py
import pickle
import numpy as np
import config
from tqdm import tqdm

keep_anno = False
only_annot = True
add_sec = False
fname = 'verify/sum_plot_annot.pkl' if only_annot else 'verify/sum_plot.pkl'


def keep(key):
    return keep_anno or (add_sec == (key[:3] == "sec" and
                                     only_annot == (key[:4] == "anno")))


images = []
for h5fn in config.h5s:
    print(h5fn)
    out = "%s has %%i that is only 0" % h5fn
    h5f = h5py.File(h5fn, 'r+')
    images.append(
        [(h5f, key, h5fn) for key in h5f.keys(
        ) if keep(key)])

images = sum(images, [])

sums = []
for h5f, gname, fn in tqdm(images):
    sums.append((np.sum(h5f[gname]), gname, fn))
with open(fname, 'wb') as f:
    pickle.dump(sums, f)
