import config
import glob
import read_tiff
from imageio import imread
import h5py


for full_fname in glob.glob(config.kmeans_path + "**/*.png", recursive=True):
    print(full_fname)
    break
