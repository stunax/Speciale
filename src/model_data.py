from __future__ import print_function
import numpy as np
import config
import re
import random
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import OneHotEncoder
from scipy import ndimage
from scipy import misc
import time
import gc

axis = (0, 1, 2)


class Model_data(object):

    def __init__(
        self, kernel_size, step=(1, 1, 1), border="same",
            debug=False, reshape=True, preprocess=False, bag_size=4,
            flat_features=False, one_hot=False,
            from_h5=False, remove_unlabeled=True, median_time=0,
            annotation_groupname=config.annotation_groupname,
            histogram=0, normalize_wieghtshare=False,
            prioritize_close_background=20, samples=0,
            augment=False, negative=-1, ignore_annotations=False):
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
            [2], sparse=False).fit(np.arange(2).reshape((2, 1)))
        self.normalize_wieghtshare = normalize_wieghtshare
        self.augment = augment
        self.negative = negative
        self.prioritize_close_background = prioritize_close_background
        self.ignore_annotations = ignore_annotations
        self.samples = samples

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
        if np.any(np.isclose(image.std(), 0)):
            return image

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
            if self.debug:
                print("Bordered image size: " + str(image.shape))
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

        if self.kernelsize[0] % 2 == 0:
            view = view[:-1]

        if self.kernelsize[1] % 2 == 0:
            view = view[:, :-1]

        if self.debug:
            print(self.kernelsize)
            print("windowed image size: " + str(image.shape))

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

    def images_to_patches(self, images):
        '''
        concat images into n * features array'''
        return [
            image.reshape(
                (image.shape[0] * image.shape[1] * images[0].shape[2],) +
                image.shape[3:])
            for image in images]

    def concat_images(self, images):
        if len(images) == 1:
            return images[0]
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
            images.append(np.array(image))
            if not self.ignore_annotations:
                annotation = h5[self.annotation_groupname + gname]
                annotations.append(np.array(annotation))

        if self.ignore_annotations:
            annotations = None

        return images, annotations

    def as_iter(self, data):
        return model_data_iter(self, data)

    def as_batcher(self, data, batchSize, max_n=99999999999999999,
                   input_target=False, wait_for_load=False):
        return model_data_batcher(
            data, batchSize, self, max_n,
            input_target=input_target, wait_for_load=wait_for_load)

    def get_rotations_2d(self, img, kernelsize):
        return [np.rot90(img, i).reshape(
            (1,) + kernelsize) for i in range(4)]

    def shuffle_res(self, images, annotations):
        indices = np.random.permutation(images.shape[0])
        return images[indices], annotations[indices]

    def augment_images(self, images, annotations):
        '''
            Rotate features.
            Features are already "flattened" so they do not need to be rotated
        '''
        new_images = []
        new_annots = []
        n = images.shape[0]
        for i in range(n):
            new_images.extend(
                self.get_rotations_2d(
                    images[i].reshape(
                        images.shape[1:-1]), self.kernelsize
                )
            )
            new_annots.extend(
                [annotations[i].reshape((1,) + annotations[i].shape)] * 4
            )
        new_images = np.concatenate(new_images, axis=0)
        new_annots = np.concatenate(new_annots, axis=0)

        return self.shuffle_res(new_images, new_annots)

    def preprocess_images(self, images):
        result = []
        for image in images:
            result.append(self.preprocess_image(image))
            gc.collect()
        return result

    def find_closes(self, annotations):
        result = []
        for annotation in annotations:
            result.append(self.find_close(annotation))
            gc.collect()
        return result

    def reshape_images(self, images):
        result = []
        for image in images:
            result.append(self.reshape_image(image))
            gc.collect()
        return result

    def samples_func(self, images, annotations):
        new_images = []
        new_annotations = []
        for i in range(len(images)):
            image = images[i]
            annotation = None if annotations is None else annotations[i]
            image, annotation = self.sample(image, annotation)
            new_images.append(image)
            if annotations is not None:
                new_annotations.append(annotation)
        return new_images, new_annotations

    def sample(self, images, annotations):
        index = np.random.choice(
            np.prod(images.shape[:2]), self.samples, replace=False)
        mask = np.zeros(np.prod(images.shape[:2]), dtype=np.bool)
        mask[index] = True
        mask.shape = images.shape[:2]
        images = images[mask]
        images.shape = (images.shape[0],) + images.shape[2:]
        if annotations is not None:
            annotations = annotations[mask]
            annotations.shape = (images.shape[0],)
        return images, annotations

    def use_annotations(self, annotations):
        return not self.ignore_annotations and (
            annotations is not None and annotations)

    def replace_close_grp(self, image, annotation, min_n, counts):
        # Replace half of the patches from far away, with samples from
        # close. If there are fewer samples close, than far away, then
        # only replace the amount that are actually close. remove temp
        # class afterwards
        # Temp class is the last one already
        n = min(
            int(min_n / 2),
            counts[1][-1])
        mask_close = annotation == config.find_close_group
        mask_far = annotation == self.negative

        idx_close = np.arange(image.shape[0])[mask_close]
        idx_close = np.random.choice(idx_close, min(n, min_n), replace=False)

        idx_far = np.arange(image.shape[0])[mask_far]
        idx_far = np.random.choice(idx_far, min(n, min_n), replace=False)

    def _do_normalize_weightshare(self, image, annotation, prod_size):
        counts = np.unique(annotation, return_counts=True)
        min_n = counts[1][np.argmax(counts[0] == 1)]
        if self.samples > 0:
            min_n = min(min_n, self.samples)
        ids = []

        for i in counts[0]:
            if i == 0:
                # these are unlabeled
                continue
            mask = (annotation == i).reshape(prod_size)
            n = np.sum(mask)
            idx = np.arange(prod_size)[mask]
            idx = np.random.choice(idx, min(n, min_n), replace=False)

            # Because we already have a view, we cannot just reshape image
            ids.append(idx)

        if self.prioritize_close_background:
            # replace some of the ids of background with close class
            n = min(min_n, ids[-1].shape[0])
            ids[0][:n] = ids[-1][:n]
            # remove close class
            ids.pop()
            annotation[annotation == config.find_close_group] = self.negative

        overall_mask = np.zeros(prod_size, dtype=np.bool)
        for idx in ids:
            overall_mask[idx] = True

        overall_mask.shape = config.image_size
        image = image[overall_mask]
        image.shape = (image.shape[0],) + image.shape[2:]
        annotation = annotation[overall_mask].reshape(image.shape[0], 1)
        return image, annotation

    def do_normalize_wieghtshare(self, images, annotations):
        image_list = []
        annotation_list = []
        prod_size = np.prod(config.image_size)
        for i in range(len(images)):
            image, annotation = self._do_normalize_weightshare(
                images[i], annotations[i], prod_size)
            image_list.append(image)
            annotation_list.append(annotation)
        return image_list, annotation_list

    def find_close(self, annotations):
        old_shape = annotations.shape
        annotations.shape = config.image_size
        foreground = annotations == 1
        struct1 = ndimage.generate_binary_structure(2, 1)
        dilation = ndimage.binary_dilation(
            foreground, structure=struct1,
            iterations=self.prioritize_close_background)
        background = annotations == -1
        close_background = np.logical_and(dilation, background)
        annotations[close_background] = config.find_close_group
        return annotations.reshape(old_shape)

    def remove_unlabeled_func(self, images, annotations):
        # Functions assumes annotations is not None
        new_images = []
        new_annotations = []
        for i in range(len(images)):
            mask = (annotations[i] != 0)
            new_images.append(images[i][mask])
            new_annotations.append(annotations[i][mask])
        return new_images, new_annotations

    def fix_negative(self, annotations):
        res = []
        for i in range(len(annotations)):
            annotation = annotations[i]
            annotation[annotation < 0] = self.negative
            res.append(annotation)
        return res

    def _handle_images(self, images, annotations=None):

        if self.from_h5:
            images, annotations = self.load_h5(images)
            gc.collect()

        if self.preprocess:
            images = self.preprocess_images(images)

        if self.use_annotations(annotations) and \
                self.normalize_wieghtshare:

            if self.prioritize_close_background:
                annotations = self.find_closes(annotations)

        if self.reshape:
            images = self.reshape_images(images)

        if self.remove_unlabeled:
            if self.normalize_wieghtshare:
                images, annotations = self.do_normalize_wieghtshare(
                    images, annotations)
            else:
                images, annotations = self.remove_unlabeled_func(
                    images, annotations)
        elif self.samples:
            images, annotations = self.samples_func(images, annotations)
        else:
            images = self.images_to_patches(images)
            if self.use_annotations(annotations):
                annotations = self.images_to_patches(annotations)

        images = self.concat_images(images)

        if self.use_annotations(annotations):
            annotations = self.concat_images(annotations)
            # annotations = annotations.ravel().astype(np.int)

            if self.negative != -1:
                annotations[annotations < 0] = self.negative

            if self.one_hot:
                # Make this change in case it's not done
                if self.negative == -1:
                    annotations[annotations < 0] = 0
                annotations = self.one_hot_encoder.transform(annotations)

        if self.histogram and self.reshape and self.flat_features:
            images = self.histogram_features(images)
        elif self.augment:
            images, annotations = self.augment_images(images, annotations)

        if len(images.shape) > 4:
            new_shape = (
                images.shape[0],) + images.shape[2:-2] + (
                np.prod(images.shape[-2:]),)
            images.shape = new_shape

        return images, annotations

    def handle_images(self, images, annotations=None):
        if self.from_h5 and self.bag_size > 1:
            bag_size = self.bag_size
            self.bag_size = 1
            images = [self._handle_images(images)
                      for i in range(max(len(images), bag_size))]
            annotations = np.concatenate([anno for _, anno in images], axis=0)
            images = np.concatenate([image for image, _ in images], axis=0)

            self.bag_size = bag_size
        else:
            images, annotations = self._handle_images(images, annotations)
        return images, annotations

    def get_pred_version(self):
        import copy
        res = copy.deepcopy(self)
        res.annotation_groupname = ""
        res.normalize_wieghtshare = False
        res.augment = False
        res.remove_unlabeled = False
        res.bag_size = 1
        res.one_hot = False
        res.samples = 0
        return res


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

    def __init__(self, data, batchSize, data_model, max_n, input_target=False,
                 wait_for_load=False):
        self.h5data = data
        self.batchSize = batchSize
        self.data_model = data_model
        self.max_n = max_n
        self.epoch = 0
        self.n = 0
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.input_target = input_target
        # self.executor = ThreadPoolExecutor(max_workers=999999999)
        self.reset_iter()
        # self.start_threads()
        # if wait_for_load:
        #     self.wait_for_result()

    def wait_for_result(self):
        while self.next_data.running():
            time.sleep(1)

    def reset_n(self):
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.n = min(self.max_n * self.batchSize, self.data[0].shape[0])

    def reset_iter(self):
        self.iter = self.data_model.as_iter(self.h5data)
        self.next_iter()

    def start_threads(self):
        def thread_func(x):
            return x.__next__()
        # start new thread
        self.next_data = self.executor.submit(thread_func, self.iter)

    def join_threads(self):
        # join old thread
        self.data = self.next_data.result()
        # Start new thread to fill while doing other stuff
        self.start_threads()

    def next_iter(self):
        # self.join_threads()
        self.data = self.iter.__next__()
        gc.collect()
        self.reset_n()

    def _next_batch(self):
        while self.batchStopIndex + self.batchSize >= self.n:
            self.next_iter()

        self.batchStartIndex = self.batchStopIndex
        self.batchStopIndex = self.batchStartIndex + self.batchSize
        if self.input_target:
            res = self.data[0][self.batchStartIndex:self.batchStopIndex]
            return [res, res]
        return [dat[self.batchStartIndex:self.batchStopIndex
                    ] for dat in self.data]

    def next_batch(self):
        try:
            return self._next_batch()
        except StopIteration:
            self.reset_iter()
            self.epoch += 1
            return self.next_batch()

    def __len__(self):
        return len(self.h5data) * config.len_settings

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.next_batch()


if __name__ == "__main__":
    n = 3

    image = np.arange((n**3) * 3).reshape((n, n, n, 3))
    annotations = np.arange((n**3)).reshape((n, n, n))

    kernelsize = (3, 3, 3)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0)
    image2 = m.handle_images([image], [annotations[:, :, 0]])

    kernelsize = (2, 3, 3)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0)
    image2 = m.handle_images([image], [annotations])

    kernelsize = (3, 2, 3)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0)
    image2 = m.handle_images([image], [annotations])

    kernelsize = (2, 2, 2)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0)
    image2 = m.handle_images([image], [annotations])

    kernelsize = (2, 2, 3)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0)
    image2 = m.handle_images([image], [annotations])

    kernelsize = (5, 7, 3)
    m = Model_data(kernel_size=kernelsize, border="same",
                   debug=True, median_time=0, histogram=10)
    image2 = m.handle_images([image], [annotations])

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
