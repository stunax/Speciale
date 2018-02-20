import numpy as np
import config
import model_data
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy.ndimage import imread


score = "f1"
score = "acc"


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
    res = np.zeros(image.shape[:2] + (1, 1))
    res[foreground] = 1
    res[background] = -1
    return res


def load_images():
    data_path = config.imag_path
    x_shape = (1024, 1024, 1, 1)
    train_X = imread(data_path + "train.png").reshape(x_shape)
    test_X = imread(data_path + "test.png").reshape(x_shape)
    test_y = make_annotation(data_path + "plane_124_18.png")
    test_y_filled = make_annotation(data_path + "plane_124_18_filled.png")
    test_y_filled[test_y == -1] = -1
    test_y_filled[test_y == 1] = 1
    train_y = make_annotation(data_path + "plane_135_14.png")

    return [train_X], [test_X], [train_y], [test_y], [test_y_filled]


train_X, test_X, train_y, test_y, test_y_filled = load_images()


def report(bag_size, kernel_size, median_filter):
    print("Bagsize:", bag_size)
    print("kernel_size:", kernel_size)
    print("median_filter:", median_filter)


def run(median_filter, kernel_size, bag_size):
    report(bag_size, kernel_size, median_filter)

    datam = model_data.Model_data(
        kernel_size=kernel_size, preprocess=False, one_hot=False,
        flat_features=True, bag_size=bag_size, median_time=median_filter)

    Xtrain, ytrain = datam.handle_images(train_X, train_y)
    Xtest, ytest = datam.handle_images(test_X, test_y)
    Xtest_filled, ytest_filled = datam.handle_images(test_X, test_y_filled)

    print("%i %i" % (ytest.size, ytest_filled.size))
    print("%f %f" % (ytest.size / np.sum(ytest == 1),
                     ytest_filled.size / np.sum(ytest_filled == 1)))

    model = RandomForestClassifier(
        n_jobs=10, max_depth=None, max_features="log2", n_estimators=2**10)
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    y_pred_filled = model.predict(Xtest_filled)
    if score == "f1":
        print("f1 score test is %f" % f1_score(ytest, y_pred))
        print("f1 score test filled is %f" %
              f1_score(ytest_filled, y_pred_filled))
    else:
        print("Accuracy score test is %f" % accuracy_score(ytest, y_pred))
        print("Accuracy score test filled is %f" %
              accuracy_score(ytest_filled, y_pred_filled))
        '''
        Accuracy score test is 0.980311
        Accuracy score test filled is 0.972167'''


for kernel_size in config.kernel_size:
    run(0, kernel_size, 1)
