import numpy as np


def cat_image_to_gray(image):
    shape = image.shape
    image = image.flatten()
    categories = np.unique(image)
    types = int(255. / categories.shape[0])
    for i in range(categories.shape[0]):
        image[image == categories[i]] = i * types
    return image.reshape(shape)
