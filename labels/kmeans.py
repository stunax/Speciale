import numpy as np
from imageio import imread, imwrite
from sklearn.cluster import KMeans
from model_data import Model_data
import glob

path = 'labels/imgs/'
im_reg = '*.png'

images = []
im_paths = []

for im in glob.glob(path + im_reg):
    # if len(images) > 1:
    #     break
    if "anno" not in im:
        images.append(imread(im)[:, :, 0].reshape(1024, 1024, 1, 1))
        im_paths.append(im)


n = len(images)

datam = Model_data(kernel_size=(11, 11, 1), flat_features=True)

Xtrain, ytrain = datam.handle_images(images)
sub_Xtrain = Xtrain[np.random.choice(Xtrain.shape[0], 3000000, False)]
print(Xtrain.shape)
print(n)

model = KMeans(n_clusters=16, n_jobs=6, verbose=0)
model.fit(sub_Xtrain)
pred = model.predict(Xtrain)
pred = pred.reshape((n, 1024, 1024))
for i in range(pred.shape[0]):
    fname = "%s_kmeans_anno.png" % (im_paths[i])
    imwrite(fname, pred[i])
