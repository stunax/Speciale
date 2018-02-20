import numpy as np
import config
import model_data
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.ndimage import imread


def make_annotation(path):
    image = imread(path)
    foreground = np.logical_and(np.logical_and(
        image[:, :, 0] == 0,
        image[:, :, 1] == 255),
        image[:, :, 2] == 0)
    background = np.logical_and(np.logical_and(
        image[:, :, 0] == 0,
        image[:, :, 1] == 255),
        image[:, :, 2] == 255)
    res = np.zeros(image.shape[:2])
    res[foreground] = 1
    res[background] = -1
    return res


def load_images():
    data_path = config.imag_path
    train_X = imread(data_path + "train.png")
    test_X = imread(data_path + "test.png")
    test_y = imread(data_path + "test_anno.png")
    test_y_filled = make_annotation(data_path + "plane_124_18_filled.png")
    train_y = make_annotation(data_path + "plane_35_14.png")

    return train_X, test_X, train_y, test_y, test_y_filled


train_X, test_X, train_y, test_y, test_y_filled = load_images()


def report(bag_size, kernel_size, median_filter):
    print("Bagsize:", bag_size)
    print("kernel_size:", kernel_size)
    print("median_filter:", median_filter)


def run(median_filter, kernel_size, bag_size):
    report(bag_size, kernel_size, median_filter)

    datam = model_data.Model_data(
        kernel_size=kernel_size, preprocess=False,
        bag_size=bag_size, median_time=median_filter)

    Xtrain, ytrain = datam.handle_images(train_X, train_y)
    Xtest, ytest = datam.handle_images(test_X, test_y)
    Xtest_filled, ytest_filled = datam.handle_images(test_X, test_y_filled)

    model = RandomForestClassifier(
        n_jobs=10, max_depth=None, max_features="log2", n_estimators=2**10)
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    print("f1 score test is %f" % f1_score(ytest, y_pred))
    y_pred = model.predict(Xtest_filled)
    print("f1 score test filled is %f" % f1_score(ytest_filled, y_pred))


for kernel_size in config.kernel_size:
    run(0, kernel_size, 1)
