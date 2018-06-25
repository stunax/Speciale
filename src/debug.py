import numpy as np
import h5py
import config
from tqdm import trange
from imageio import imwrite


def dnn_pred_image(data, sess, op, X):
    preds = []
    for i in trange(int(data.shape[0] / config.batch_size)):
        from_i = i * config.batch_size
        to_i = min(data.shape[0], (i + 1) * config.batch_size)
        batch = data[from_i:to_i].reshape((-1,) + config.patch_size)
        preds.append(sess.run(op, feed_dict={X: batch}))
    return np.concatenate(preds, axis=0)


def h5_file_name(h5file):
    name = h5file.filename
    return name.rsplit("/", 1)[1]


def pred_image(x):
    h5s, pred_classes, X, sess, data_model, data_model2 = x
    print(h5s)
    if h5s[1] not in h5s[0]:
        return
    data, _ = data_model.handle_images([h5s])
    n = np.sum(data)
    if not n:
        print("EMPTY")
        return
    pred = dnn_pred_image(data, sess, pred_classes, X)
    pred = pred.reshape((1024, 1024))
    print(np.unique(pred, return_counts=True))
    pred[pred == 1] = 255
    imwrite(config.test_imag_path +
            h5_file_name(h5s[0]) + h5s[1] + "_pred.png", pred)
    data, _ = data_model2.handle_images([h5s])
    data = data.reshape((1024, 1024)).astype(np.uint8)
    imwrite(config.test_imag_path +
            h5_file_name(h5s[0]) + h5s[1] + "_org.png", data)


def _get_near(t, z, tp, zp):
    return (str(t + tp).zfill(3), str(z + zp).zfill(2))


def get_near(h5s, pred_classes, X, sess, data_model, data_model2):
    nears = []
    ps = [-1, 0, 1]
    for h5, df in h5s:
        t, z = map(int, df.split("_"))
        for tp in ps:
            for zp in ps:
                nears.append(((h5, "%s_%s" % _get_near(t, z, tp, zp)),
                              pred_classes, X, sess, data_model, data_model2))
    return nears


def pred_image2(x, args):
    h5s, model, data_model, data_model2 = x
    print(h5s)
    if h5s[1] not in h5s[0]:
        return
    data, _ = data_model.handle_images([h5s])
    n = np.sum(data)
    if not n:
        print("EMPTY")
        return
    pred = model.predict(data, batch_size=config.batch_size)
    pred = np.argmax(pred, axis=1)
    pred = pred.reshape((1024, 1024)).astype(np.uint8)
    pred[pred == 1] = 255
    imwrite(config.test_imag_path +
            h5_file_name(h5s[0]) + h5s[1] + "%i_%i_%i_pred.png" % (
                args.normalize, args.median_time, args.close_size), pred)
    data, _ = data_model2.handle_images([h5s])
    data = data.reshape((1024, 1024)).astype(np.uint8)
    imwrite(config.test_imag_path +
            h5_file_name(h5s[0]) + h5s[1] + "_org.png", data)


def get_near2(h5s, model, data_model, data_model2):
    nears = []
    ps = [-1, 0, 1]
    for h5, df in h5s:
        t, z = map(int, df.split("_"))
        for tp in ps:
            for zp in ps:
                nears.append(
                    ((h5, "%s_%s" % _get_near(t, z, tp, zp)),
                     model, data_model, data_model2))
    return nears


def test_model(model, data_model, args, pred_type):
    data_model = data_model.get_pred_version()
    h5file = config.h5s[2]
    h5f = h5py.File(h5file)
    h5 = [(h5f, config.test_image)]

    data = data_model.handle_images(h5)
    h5 = h5[0]

    pred = model.predict(data[0], batch_size=config.batch_size)
    pred = np.argmax(pred, axis=1)
    pred = pred.reshape((1024, 1024)).astype(np.uint8)
    pred[pred == 1] = 255

    imwrite(config.test_imag_path +
            h5_file_name(h5[0]) + h5[1] + "%s_%i_%i_%i_pred.png" % (
                pred_type, args.normalize, args.median_time, args.close_size
            ), pred)
