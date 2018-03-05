from __future__ import print_function
import numpy as np
import config
import re
import random
from random import shuffle
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import OneHotEncoder
from scipy import misc

axis = (0, 1, 2)


class Model_data(object):

    def __init__(
        self, kernel_size, step=(1, 1, 1), border="same",
            debug=False, reshape=True, preprocess=False, bag_size=4,
            flat_features=False, one_hot=True,
            from_h5=False, remove_unlabeled=True, median_time=0,
            annotation_groupname=config.annotation_groupname,
            histogram=0, normalize_wieghtshare=False, augment=False):
        self.debug = debug
        self.kernelsize = kernel_size
        self.step = step
        self.border = border
        self.reshape = reshape
        self.preprocess = preprocess
        self.from_h5 = from_h5
        self.remove_unlabeled = remove_unlabeled
        self.median_time = median_time
        self.bag_size = bag_size
        self.flat_features = flat_features
        self.annotation_groupname = annotation_groupname
        self.histogram = histogram
        self.one_hot = one_hot
        self.one_hot_encoder = OneHotEncoder(
            [3], sparse=False).fit(np.arange(3).reshape((3, 1)))
        self.normalize_wieghtshare = normalize_wieghtshare
        self.augment = augment

    def bordersize(self):
        return (
            int((self.kernelsize[0]) / 2),
            int((self.kernelsize[1]) / 2),
            int((self.kernelsize[2]) / 2),
        )

    def same_border_image(self, image):
        '''
        Handles same border image'''
        bordersize = self.bordersize()

        picbordered = np.zeros((
            image.shape[0] + bordersize[0] * 2,
            image.shape[1] + bordersize[1] * 2,
            image.shape[2],
            # image.shape[2] + bordersize[2] * 2,
            image.shape[3]))
        picbordered[
            bordersize[0]:(image.shape[0] + bordersize[0]),
            bordersize[1]:(image.shape[1] + bordersize[1]),
            :,
            # bordersize[2]:(image.shape[2] + bordersize[2]),
            :] = image
        return picbordered

    def preprocess_image(self, image):
        '''
        Handles preprocessing of individual images
        Images are still 3d at this point'''
        # Return image, if no std
        # if np.any(np.isclose(image.std(axis=axis), 0)):
        #     return image

        # Always normalize image around 0 with 1 std
        image = image.astype(np.float)
        image = (image - image.mean(axis=axis)) / image.std(axis=axis)
        # Veryfy mean and std
        if self.debug:
            print("Image mean: " + np.array_str(
                image.mean(axis=axis), precision=2))
            print("Image std: " + np.array_str(
                image.std(axis=axis), precision=2))

        return image

    def reshape_image(self, image):
        '''
        Reshape image to 3d * features instead of 3d + channels images'''
        if self.border == "same":
            image = self.same_border_image(image)
        elif self.border == "valid":
            image = image
        else:
            raise TypeError("Does not know this specific border type")

        # Create view into bordered image, without copying data,
        # that has kernselsize
        view = view_as_windows(
            image,
            window_shape=self.kernelsize + (image.shape[-1],),
            # step=self.step + (image.shape[-1],)
        )
        # Flatten view
        if self.flat_features:
            view = view.reshape(
                view.shape[0],
                view.shape[1],
                view.shape[2],
                view.shape[-1] * np.prod(self.kernelsize))

        if self.debug:
            print("Image view shape: " + str(view.shape))

        return view

    def histogram_features(self, image):
        # return image
        if self.debug:
            print(image.shape)

        def histfunc(entry):
            return np.histogram(
                entry, bins=self.histogram, range=(0, 255))[0]

        image = np.apply_along_axis(histfunc, 1, image)
        if self.debug:
            print(image.shape)
        return image

    def concat_images(self, images):
        '''
        concat images into n * features array'''
        images = [
            image.reshape(
                (image.shape[0] * image.shape[1] * images[0].shape[2],) +
                image.shape[3:])
            for image in images]
        data = np.concatenate(images, axis=0)
        return data

    def load_image(self, path):
        img = misc.imread(path)
        return img

    def load_images(self, paths):
        images = [self.load_image(path) for path in paths]

        return images

    def median_filter(self, h5, im_t, im_z):
        filter_image = np.zeros(
            config.image_size + (2 * self.median_time + 1, config.nchannels))
        # filter_image = np.array(filter_image_size)
        im_t = int(im_t)
        i = 0
        for t in range(-self.median_time, self.median_time + 1):
            grp_name = config.groupname_format % (str(im_t + t).zfill(3), im_z)
            # Could be outside image. Just let it be 0's
            if grp_name in h5:
                filter_image[:, :, i, :, np.newaxis] = h5[grp_name]
            i += 1

        return np.median(filter_image, axis=2)

    def load_slice(self, h5, gname):
        n_slices = self.kernelsize[2]
        image_slice = np.zeros(config.image_size +
                               (n_slices, config.nchannels))

        im_t, im_z = re.findall("[0-9]+", gname)

        for i in range(n_slices):
            z = str(int(im_z) - int(self.kernelsize[2] / 2.))
            # Just call median filter even though it wont do anything. Should
            # not impact performance
            image_slice[:, :, i, :] = self.median_filter(h5, im_t, z)
        return image_slice

    def load_h5(self, h5, ignore_bag=False):
        h5 = [x for x in h5 if self.annotation_groupname + x[1] in x[0]]
        # Ignore_bag is used when scoring.
        if not ignore_bag:
            h5 = random.sample(h5, min(self.bag_size, len(h5)))
        images = []
        annotations = []
        # As loop due to advanced functionality
        for h5, gname in h5:
            image = self.load_slice(h5, gname)

            annotation = h5[self.annotation_groupname + gname]

            images.append(np.array(image))
            annotations.append(np.array(annotation))

        return images, annotations

    def as_iter(self, data):
        return model_data_iter(self, data)

    def as_batcher(self, data, batchSize):
        return model_data_batcher(data, batchSize, self)

    def do_normalize_wieghtshare(self, images, annotations):
        counts = np.unique(annotations, return_counts=True)
        min_n = min(counts[1])
        image_list = []
        annotation_list = []
        for i in counts[0]:
            mask = annotations == i
            idx = np.random.choice(np.sum(mask), min_n, replace=False)
            image_list.append(images[mask][idx])
            annotation_list.append(annotations[mask][idx])
        images = np.concatenate(image_list)
        annotations = np.concatenate(annotation_list)
        return images, annotations

    def get_rotations_2d(self, img):
        return [np.rot90(img, i, (0, 1)) for i in range(4)]

    def augment_images(self, images, annotations):
        new_images = []
        new_annots = []

        for i in range(len(images)):
            new_images += self.get_rotations_2d(
                images[i].reshape(images.shape[2:-1]))
            new_annots += [annotations[i]] * 4
        shuffle(new_images)
        shuffle(new_annots)
        new_images = np.concatenate(new_images, axis=0)
        new_annots = np.concatenate(new_annots, axis=0)

        return new_images, new_annots

    def handle_images(self, images, annotations=None):

        if self.from_h5:
            images, annotations = self.load_h5(images)

        if self.preprocess:
            images = [self.preprocess_image(image) for image in images]

        if self.reshape:
            images = [self.reshape_image(image) for image in images]

        images = self.concat_images(images)

        if self.histogram and self.reshape:
            images = self.histogram_features(images)

        if annotations is not None and annotations:
            annotations = self.concat_images(
                annotations).ravel().astype(np.int)
            # # Remove entries, with 0 class
            if self.remove_unlabeled:
                mask = (annotations != 0)
                images = images[mask]
                annotations = annotations[mask]
                if self.normalize_wieghtshare:
                    images, annotations = self.do_normalize_wieghtshare(
                        images, annotations)
            if self.one_hot:
                annotations[annotations == -1] = 2
                annotations = self.one_hot_encoder.transform(
                    annotations.reshape(annotations.shape + (1,)))

        if self.augment:
            images, annotations = self.augment_images(images, annotations)

        return images, annotations


class model_data_iter(object):
    """docstring for model_data_iter"""

    def __init__(self, data_model, data):
        super(model_data_iter, self).__init__()
        self.data_model = data_model
        self.data = random.sample(data, len(data))
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < len(self.data):
            res = self.data_model.handle_images(
                self.data[self.num:(self.num + self.data_model.bag_size)])
            self.num += self.data_model.bag_size
            return res
        else:
            raise StopIteration()

    def next(self):
        return self.__next__()


class model_data_batcher:
    'Splits data into mini-batches'

    def __init__(self, data, batchSize, data_model):
        self.h5data = data
        self.batchSize = batchSize
        self.data_model = data_model
        self.reset_iter()
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.reset_n()

    def reset_n(self):
        self.n = self.data[0].shape[0]

    def reset_iter(self):
        self.iter = self.data_model.as_iter(self.h5data)
        self.data = self.iter.__next__()

    def _next_batch(self):
        if self.batchStopIndex == self.n:
            self.data = self.iter.__next__()
            self.batchStartIndex = 0
            self.batchStopIndex = 0
            self.reset_n()

        self.batchStartIndex = self.batchStopIndex % self.noData
        self.batchStopIndex = min(
            self.batchStartIndex + self.batchSize, self.n)
        # X = self.data[0][self.batchStartIndex:self.batchStopIndex]
        # y = self.data[1][self.batchStartIndex:self.batchStopIndex]
        # return X, y
        return [dat[self.batchStartIndex:self.batchStopIndex
                    ] for dat in self.data]

    def next_batch(self):
        try:
            return self._next_batch()
        except StopIteration:
            self.reset_iter()
            return self._next_batch()


if __name__ == "__main__":
    n = 10
    kernelsize = (5, 7, 3)
    image = np.arange((n**3) * 3).reshape((n, n, n, 3))
    annotations = np.arange((n**3)).reshape((n, n, n))
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0, histogram=10)
    image2 = m.handle_images([image])

    # Data is 3d with 3 channels
    for m in [0, 1]:
        annotations[0, :] = 0
        m = Model_data(kernel_size=kernelsize, border="same",
                       debug=True, median_time=m)
        image2 = m.reshape_image(image)
        m = Model_data(kernel_size=kernelsize, border="valid", debug=True)
        image3 = m.reshape_image(image)
        images = m.concat_images([image2, image3])
        print(images.shape)

        m = Model_data(kernel_size=kernelsize, border="same",
                       debug=True, median_time=m, remove_unlabeled=True)
        image2, anno = m.handle_images([image], [annotations])

        m = Model_data(kernel_size=kernelsize, border="same",
                       debug=True, median_time=m, remove_unlabeled=False)
        image3, anno = m.handle_images([image], [annotations])
        print(image2.shape, image3.shape)
