import config
import glob
import h5py
import random
import shutil

h5s = {str(fn): h5py.File(config.data_path + str(fn) + ".h5", "a")
       for fn in range(1, 6)}

files = []
for full_fname in glob.glob(
        config.kmeans_path + "**/*_pred.png", recursive=True):
    _, h5i, fname_pred = full_fname.rsplit("/", 2)
    fname = fname_pred[:6]
    h5 = h5s[h5i]
    if config.c_anno_groupname + fname in h5:
        continue
    files.append((h5i, fname))

print(len(files))

files = random.sample(files, 10)


def copy(h5i, fname):
    copy_path = "%s%s/%s" % (config.kmeans_path, h5i, fname)
    target_path = "%s%s.h5%s" % (config.annotations_path_prep, h5i, fname)
    shutil.copyfile(copy_path, target_path)


for h5i, name in files:
    ext = ".png"
    pred_name = "%s_pred%s" % (name, ext)
    fname = "%s%s" % (name, ext)
    copy(h5i, pred_name)
    copy(h5i, fname)
