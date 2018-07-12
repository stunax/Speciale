from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import Adam
import config
from model_data import Model_data
from sklearn.model_selection import train_test_split
from keras import callbacks
import debug
import os

leaky_alpha = 0.3
augment = True


def make_model(args):

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',
                            input_shape=config.patch_size_simple,
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.dropout))
    model.add(Convolution2D(64, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.dropout))

    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=regularizers.l2(
        0.01), activation='relu',))
    model.add(Dropout(args.dropout))
    model.add(Dense(config.num_classes, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.01)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(args.learning_rate),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    args = config.get_args("simple")
    # print(args)

    model = make_model(args)

    weights_path = config.weights_path % (
        "simple", args.normalize, args.median_time, augment, args.close_size)

    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (0.5 ** epoch))
    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='auto')
    checkpoint = callbacks.ModelCheckpoint(
        weights_path,
        monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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

    X_train, X_test = train_test_split(
        h5s, test_size=0.2, random_state=args.random_state)
    X_train, X_val = train_test_split(
        X_train, test_size=0.2, random_state=args.random_state)

    model.summary()

    if config.use_saved_weights and os.path.isfile(weights_path):
        print("loading weights")
        model.load_weights(weights_path)

    if config.train:
        X_train_batcher = data_model.as_batcher(
            X_train, config.batch_size, 9999999)  # config.len_settings)
        X_val_batcher = data_model.as_batcher(
            X_val, config.batch_size, 9999999)  # config.len_settings)

        model.fit_generator(
            X_train_batcher, steps_per_epoch=len(
                X_train) * config.len_settings * 4**augment,
            epochs=config.max_epochs, verbose=1,
            callbacks=[tb, lr_decay, earlyStopping, checkpoint],
            validation_data=X_val_batcher,
            validation_steps=int(
                len(X_val) * config.len_settings) * 4**augment,
            class_weight=None,
            max_queue_size=config.max_queue_size, workers=1,
            use_multiprocessing=True,
            shuffle=True, initial_epoch=0
        )

    X_test_batcher = data_model.as_batcher(
        X_test, config.batch_size, 9999999)  # config.len_settings)

    test_res = model.evaluate_generator(
        X_test_batcher, steps=len(X_test) * config.len_settings * 4**augment,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
    )

    # Semisupervised, normalized, median filter, loss, accuracy
    # print("False,%s,%i %%,%f,%f" % (
    #     str(args.normalized), args.median_time, test_res[0], test_res[1]))

    config.print_to_result("False", str(args.normalize),
                           args.median_time, test_res[0], test_res[1],
                           args.close_size)

    debug.test_model(model, data_model, args, "simple")

    # data_model3 = data_model.get_pred_version()

    # nears = debug.get_near2(X_val, model, data_model3, data_model2)
    # for x in nears:
    #     debug.pred_image2(x, args)
