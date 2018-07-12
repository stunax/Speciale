# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras import regularizers
# from keras.optimizers import Adam
import config
from model_data import Model_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import callbacks
import numpy as np
from simple_keras import make_model

leaky_alpha = 0.3
augment = True


def get_conf_mat(model, data_model, data):
    data_model.bag_size *= 4
    X, y = data_model.handle_images(data)
    pred = model.predict(X, batch_size=config.batch_size)
    pred = np.argmax(pred, 1)
    y = np.argmax(y, 1)
    return confusion_matrix(pred, y).ravel()


if __name__ == '__main__':
    args = config.get_args("simple_keras")
    args.learning_rate *= 15
    # print(args)

    data_model = Model_data(
        config.patch_size_simple, bag_size=config.bag_size,
        preprocess=args.normalize,
        annotation_groupname=config.c_anno_groupname,
        from_h5=True, one_hot=True, median_time=args.median_time,
        normalize_wieghtshare=True, augment=augment, negative=0,
        prioritize_close_background=args.close_size)
    data_model2 = Model_data(
        (1, 1, 1), bag_size=1,
        annotation_groupname="",
        from_h5=True, one_hot=False, median_time=0,
        normalize_wieghtshare=False, augment=False, remove_unlabeled=False)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    model = make_model(args)

    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='auto')

    X_train, X_test = train_test_split(
        h5s, test_size=0.2)
    X_train, X_val = train_test_split(
        X_train, test_size=0.2)

    X_train_batcher = data_model.as_batcher(
        X_train, config.batch_size, 9999999)  # config.len_settings)
    X_val_batcher = data_model.as_batcher(
        X_val, config.batch_size, 9999999)  # config.len_settings)

    model.fit_generator(
        X_train_batcher, steps_per_epoch=len(
            X_train) * config.len_settings * 4**augment,
        epochs=10, verbose=1,
        callbacks=[tb, earlyStopping],
        validation_data=X_val_batcher,
        validation_steps=int(
            len(X_val) * config.len_settings),
        class_weight=None,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
        shuffle=True, initial_epoch=0
    )

    X_train_batcher = None
    X_val_batcher = None

    X_test_batcher = data_model.as_batcher(
        X_test, config.batch_size, 9999999)  # config.len_settings)

    test_res = model.evaluate_generator(
        X_test_batcher, steps=len(
            X_test) * config.len_settings * 4**augment,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
    )
    X_test_batcher = None
    model = None
    config.print_to_result("False", str(bool(args.normalize)),
                           args.median_time, test_res[0], test_res[1],
                           close_size=0)

    print(get_conf_mat(model, data_model, X_test))
