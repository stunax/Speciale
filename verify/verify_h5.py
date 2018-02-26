import h5py
import numpy as np
import config
from tqdm import tqdm

threshold = 10000


def is_empty(h5, gname):
    data = np.array(h5[gname])

    n = np.sum(data)
    res = n < threshold
    if data.shape[0] != 1024 or data.shape[1] != 1024:
        res = True
    not_any = not np.any(data)
    # if res:
    #     print(gname)
    return res, not_any


for h5f in config.h5s:
    print(h5f)
    out = "%s has %%i that is only 0" % h5f
    h5f = h5py.File(h5f, 'r+')
    n = 0
    not_any_n = 0
    ts = []
    zs = []
    # for df in tqdm(h5f.keys()):
    for df in tqdm(h5f.keys()):
        if not (df[:3] == "sec" or df[:4] == "anno"):
            threshold, not_any = is_empty(h5f, df)
            n += threshold
            not_any_n = not_any
            if threshold or not_any:
                t, z = df.split("_")
                ts.append(int(t))
                zs.append(int(z))
    # empties = [is_empty(h5f, df) for df in h5f.keys() if df[:4] != "anno"]
    # n = np.sum(empties)
    print(n, not_any_n)
