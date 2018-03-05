import numpy as np
import config
import h5py
import pandas as pd
# import matplotlib as mpl
# mpl.use('Agg')
# import ggplot


def keep(key):
    return key[:4] == "anno"


for h5fn in config.h5s:
    print(h5fn)
    out = "%s has %%i that is only 0" % h5fn
    h5f = h5py.File(h5fn, 'r+')
    images = [(h5f, key, h5fn) for key in h5f.keys() if keep(key)]
    print(len(images))
    placement = [key[-6:].split("_") for _, key, _ in images]
    placement = list(zip(*placement))
    df = pd.DataFrame()
    df['z'] = pd.to_numeric(placement[1], errors='coerce')
    df['t'] = pd.to_numeric(placement[0], errors='coerce')
    print(np.unique(df.z, return_counts=True))
    print(np.unique(df.t, return_counts=True))

