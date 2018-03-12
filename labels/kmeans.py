from imageio import imread, imwrite
from sklearn.cluster import KMeans
from model_data import Model_data
import glob

path = 'labels/'
im_reg = '*.png'

images = []
im_paths = []

for im in glob.glob(path + im_reg):
    if "anno" not in im:
        images.append(imread(im).reshape(1024, 1024, 1, 3))
        im_paths.append(im)


n = len(images)

datam = Model_data(kernel_size=(12, 12, 1))

Xtrain, ytrain = datam.handle_images(images)
print(Xtrain.shape)

model = KMeans(n_clusters=16)
pred = model.fit_transform(Xtrain).reshape((n, 1024**2))
for i in range(pred.shape[0]):
    fname = "%s_kmeans.png" % (im_paths[i])
    imwrite(fname, pred[i])
