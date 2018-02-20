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
        image[:, :, 0] < 10,
        image[:, :, 1] > 245),
        image[:, :, 2] < 10)
    background = np.logical_and(np.logical_and(
        image[:, :, 0] < 10,
        image[:, :, 1] > 245),
        image[:, :, 2] > 245)
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
    # test_y_filled[test_y == 255] = 1
    # test_y_filled[test_y == 0] = -1
    train_y = make_annotation(data_path + "plane_135_14.png")
    train_y_filled = make_annotation(data_path + "plane_135_14_filled.png")
    # train_y_filled[train_y == 1] = 1
    # train_y_filled[train_y == -1] = -1
    print(np.unique(train_y, return_counts=True))

    return ([train_X], [test_X], [train_y],
            [test_y], [test_y_filled], [train_y_filled])


train_X, test_X, train_y, test_y, test_y_filled, train_y_filled = load_images()


def present(model, X, y, score):
    pred = model.predict(X)
    if score == "f1":
        print("f1 score test filled is %f" % f1_score(y, pred))
    else:
        print("Accuracy score test is %f" % accuracy_score(y, pred))


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
    Xtrain_filled, ytrain_filled = datam.handle_images(train_X, train_y_filled)
    Xtest, ytest = datam.handle_images(test_X, test_y)
    Xtest_filled, ytest_filled = datam.handle_images(test_X, test_y_filled)

    print("%i %i" % (ytest.size, ytest_filled.size))
    print("%f %f" % (np.sum(ytest.ravel() == 1) / ytest.size,
                     np.sum(ytest_filled.ravel() == 1) / ytest_filled.size))

    print("%i %i" % (ytrain.size, ytrain_filled.size))
    print("%f %f" % (np.sum(ytrain.ravel() == 1) / ytrain.size,
                     np.sum(ytrain_filled.ravel() == 1) / ytrain_filled.size))

    model = RandomForestClassifier(
        n_jobs=10, max_depth=None, max_features="log2", n_estimators=2**10)
    model.fit(Xtrain, ytrain)
    present(model, Xtest, ytest, score)
    present(model, Xtest_filled, ytest_filled, score)
    '''
    Accuracy score test is 0.980311
    Accuracy score test filled is 0.972167'''
    model.fit(Xtrain_filled, ytrain_filled)
    present(model, Xtest, ytest, score)
    present(model, Xtest_filled, ytest_filled, score)
    '''
    Accuracy score test is 0.980311
    Accuracy score test filled is 0.972167'''


for kernel_size in config.kernel_size:
    run(0, kernel_size, 1)
