import h5py
import numpy as np
import config
from tqdm import tqdm

threshold = 10000


def is_empty(h5, gname):
    data = h5[gname]

    n = np.sum(data)
    res = n < threshold
    if data.shape[0] != 1024 or data.shape[1] != 1024:
        res = True
    # if res:
    #     print(gname)
    return res


for h5f in config.h5s:
    print(h5f)
    out = "%s has %%i that is only 0" % h5f
    h5f = h5py.File(h5f, 'r+')
    n = 0
    # for df in tqdm(h5f.keys()):
    for df in tqdm(h5f.keys()):
        if not (df[:3] == "sec" or df[:4] == "anno"):
            n += is_empty(h5f, df)
    # empties = [is_empty(h5f, df) for df in h5f.keys() if df[:4] != "anno"]
    # n = np.sum(empties)
    print(n)
