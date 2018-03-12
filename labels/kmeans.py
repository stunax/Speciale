from imageio import imread
from sklearn.cluster import KMeans
from ..src.Model_data import Model_data
import glob

path = 'labels/'
im_reg = '*.png'

images = []

for im in glob.glob(path + im_reg):
    if "anno" not in im:
        images.append(imread(im))


n = len(images)

datam = Model_data(
    kernel_size=(12, 12, 1))

Xtrain, ytrain = datam.handle_images(images)

model = KMeans(n_clusters=16)
pred = model.fit_transform(Xtrain).reshape((n, 1024**2))
