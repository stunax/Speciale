import config
import glob
import read_tiff
from imageio import imread
import h5py

h5s = {str(fn): h5py.File(config.data_path + str(fn) + ".h5", "a")
       for fn in range(1, 6)}


for full_fname in glob.glob(config.annotations_path_c):
    # fname 4_023_19.png
    fname = full_fname.rsplit("/", 1)[1]
    h5, fname = fname[:-4].split(".", 1)
    fname = config.c_anno_groupname + fname
    if fname in h5s[h5]:
        continue
    image = imread(full_fname)
    annotation = read_tiff.get_mask(image)
    h5s[h5].create_dataset(fname, data=annotation)