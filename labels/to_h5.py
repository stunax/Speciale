import config
import glob
import read_tiff
from imageio import imread
import h5py

h5s = {str(fn): h5py.File(config.data_path + str(fn) + ".h5", "a")
       for fn in range(1, 6)}


for full_fname in glob.glob(config.annotations_path_c + "*.png"):
    # fname 4.h5023_19*.png
    fname = full_fname.rsplit("/", 1)[1]
    # h5, fname = fname[:-4].split(".", 1)
    h5 = h5s[fname[0]]
    fname = fname[4:10]
    fname = config.c_anno_groupname + fname
    if fname in h5:
        h5.__delitem__(fname)
    print (fname, h5)
    image = imread(full_fname)
    annotation = read_tiff.get_mask(image)
    h5.create_dataset(fname, data=annotation)
