from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import config
from model_data import Model_data
from sklearn.model_selection import train_test_split
from keras import callbacks
import debug

leaky_alpha = 0.3


def make_model(args):

    model = Sequential()
    model.add(Convolution2D(32, (3, 3),  # activation='relu',
                            input_shape=config.patch_size))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.dropout))
    model.add(Convolution2D(64, (3, 3)))  # activation='relu'))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.dropout))

    model.add(Flatten())
    model.add(Dense(128))  # , activation='relu'))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(args.dropout))
    model.add(Dense(config.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(args.learning_rate),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=config.max_epochs, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument(
        '--run_name', default='simple_keras/' + config.run_name)
    parser.add_argument('--learning_rate', default=config.learning_rate,
                        type=float, help="Initial learning rate")
    parser.add_argument('--normalize', default=config.normalize_input,
                        type=bool, help="normalize images?")
    parser.add_argument('--median_time', default=config.median_time, type=int,
                        help="Median time filter the data?")
    parser.add_argument('--dropout', default=config.dropout, type=int,
                        help="Dropout")
    args = parser.parse_args()
    print(args)

    model = make_model(args)

    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (0.7 ** epoch))
    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='auto')
    checkpoint = callbacks.ModelCheckpoint(
        '/home/dpj482/data/model_weights/weights{epoch:02d}-{val_acc:.2f}.h5',
        monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    data_model = Model_data(
        config.patch_size, bag_size=config.bag_size,
        preprocess=args.normalize,
        annotation_groupname=config.c_anno_groupname,
        from_h5=True, one_hot=True, median_time=args.median_time,
        normalize_wieghtshare=True, augment=True, negative=0)
    data_model2 = Model_data(
        (1, 1, 1), bag_size=1,
        annotation_groupname="",
        from_h5=True, one_hot=False, median_time=0,
        normalize_wieghtshare=False, augment=False, remove_unlabeled=False)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    X_train, X_test = train_test_split(
        h5s, test_size=0.2, random_state=config.random_state)
    X_train, X_val = train_test_split(
        X_train, test_size=0.2, random_state=config.random_state)

    X_train_batcher = data_model.as_batcher(
        X_train, config.batch_size, 9999999)  # config.len_settings)
    X_val_batcher = data_model.as_batcher(
        X_val, config.batch_size, 9999999)  # config.len_settings)

    model.fit_generator(
        X_train_batcher, steps_per_epoch=len(
            X_train) * config.len_settings * 0.5,
        epochs=config.max_epochs, verbose=1,
        callbacks=[tb, lr_decay, earlyStopping, checkpoint],
        validation_data=X_val_batcher,
        validation_steps=int(len(X_val) * config.len_settings / 3),
        class_weight=None,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
        shuffle=True, initial_epoch=0
    )

    X_test_batcher = data_model.as_batcher(
        X_test, config.batch_size, 9999999)  # config.len_settings)

    model.evaluate_generator(
        X_test_batcher, steps=len(X_test) * config.len_settings,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
    )

    nears = debug.get_near2(X_val, model, data_model, data_model2)
    map(debug.pred_image2, nears)
