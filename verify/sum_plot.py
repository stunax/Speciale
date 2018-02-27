import h5py
import pickle
import numpy as np
import config
from tqdm import tqdm

threshold = 10000
keep_anno = False


images = []
for h5f in config.h5s:
    print(h5f)
    out = "%s has %%i that is only 0" % h5f
    h5f = h5py.File(h5f, 'r+')
    images.append([(h5f, key) for key in h5f.keys() if keep_anno or (key[
        :3] == "sec" or key[:4] == "anno")])

images = sum(images, [])

sums = []
for h5f, gname in tqdm(images):
    sums.append((np.sum(h5f[gname]), gname))
with open('verify/sum_plot.pkl', 'wb') as f:
    pickle.dump(sums, f)
