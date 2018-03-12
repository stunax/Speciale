import numpy as np
import config
import h5py
import pandas as pd
import pickle

out_extra = ''
path = "verify/"


# def keep(key):
#     return len(key) == 6
#     # return key[:4] == "anno"


# for h5fn in config.h5s:
#     print(h5fn)
#     out = "%s has %%i that is only 0" % h5fn
#     h5f = h5py.File(h5fn, 'r+')
#     images = [(h5f, key, h5fn.split("/")[-1])
#               for key in h5f.keys() if keep(key)]
#     print(len(images))
#     placement = [key[-6:].split("_") for _, key, _ in images]
#     placement = list(zip(*placement))
#     df = pd.DataFrame()
#     df['z'] = pd.to_numeric(placement[1], errors='coerce')
#     df['t'] = pd.to_numeric(placement[0], errors='coerce')
#     df["fname"] = [fname for _, _, fname in images]
#     zs = np.unique(df.z)
#     ts = np.unique(df.t)
#     df['sum'] = [np.sum(h5f[df.fname.iloc[i]] for i in range(len(df.fname)))]

#     fn = out_extra + 'sum_density_z.png'
#     p = ggplot(aes(x='z', y='sum', colour='dataset'), data=df)
#     p_dens = p + geom_point() + facet_wrap('dataset', ncol=3) + xlab("z")
#     p_dens.save(path + fn)


def keep(key):
    return len(key) == 6


def match_sum(h5f, key, label):
    label_name = "anno" + key
    res = 0
    if label_name in h5f.keys():
        labels = np.array(h5f[label_name])
        res = np.sum(labels[labels == label])
    return res


for h5fn in config.h5s:
    print(h5fn)
    out = "%s has %%i that is only 0" % h5fn
    h5f = h5py.File(h5fn, 'r+')
    images = [(h5f, key, h5fn.split("/")[-1])
              for key in h5f.keys() if keep(key)]

    df = pd.DataFrame()
    df['fname'] = [x[2] for x in images]
    df['image'] = [x[1] for x in images]
    placement = [key[-6:].split("_") for _, key, _ in images]
    placement = list(zip(*placement))
    df['z'] = pd.to_numeric(placement[1], errors='coerce')
    df['t'] = pd.to_numeric(placement[0], errors='coerce')
    df['intensity_sum'] = [np.sum(h5f[x[1]]) for x in images]

    df['front_sum'] = [match_sum(h5f, x[1], 1) for x in images]
    df['back_sum'] = [match_sum(h5f, x[1], -1) for x in images]

    fname = path + images[0][2] + '.pkl'
    with open(fname, "w") as f:
        pickle.dump(df, fname)

