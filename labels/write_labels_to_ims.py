from imageio import imread, imwrite
import numpy as np
from read_tiff import convert_loaded_annotations
import glob


path = 'labels/'
im_reg = '*_new_anno.png'

for impath in glob.glob(path + im_reg):
    print(impath)
    annotations = imread(impath)
    impath = impath[:-(len(im_reg) - 1)]
    print(impath)
    image = np.array(imread(impath))

    fore, back = convert_loaded_annotations(annotations)
    image[fore] = np.array([[[0, 255, 0]]])
    image[back] = np.array([[[0, 255, 255]]])

    impath = impath[:-4] + 'with_new_anno.png'
    imwrite(impath, image)
