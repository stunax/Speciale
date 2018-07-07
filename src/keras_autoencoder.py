from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, UpSampling2D, Reshape
from keras.optimizers import Adam
from keras import regularizers
# from keras import backend as K
from keras import callbacks
import config
from model_data import Model_data
import os.path
import gc
import debug

encoded_size = 128
# patch_size = (32, 32, 5)
image_samples = 30000
onehot = True
augment = True

weights_path = config.weights_path

earlyStopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0, mode='auto')


def make_autoencoder_model(args):

    input_img = Input(shape=config.patch_size)
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    drop1 = Dropout(args.dropout)(input_img)
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop2 = Dropout(args.dropout)(pool1)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop3 = Dropout(args.dropout)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop3)

    dense0 = Flatten()(conv3)
    drop4 = Dropout(args.dropout)(dense0)
    dense1 = Dense(1024, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01))(drop4)
    drop40 = Dropout(args.dropout)(dense1)
    encoded = Dense(encoded_size, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(drop40)
    drop41 = Dropout(args.dropout)(encoded)

    # decoder
    dense2 = Dense(1024,
                   kernel_regularizer=regularizers.l2(0.01))(drop41)
    reshape1 = Reshape((4, 4, 64))(dense2)
    drop5 = Dropout(args.dropout)(reshape1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop5)
    up1 = UpSampling2D((2, 2))(conv4)
    drop6 = Dropout(args.dropout)(up1)
    conv5 = Conv2D(64, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop6)
    up2 = UpSampling2D((2, 2))(conv5)
    drop7 = Dropout(args.dropout)(up2)
    conv6 = Conv2D(32, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(drop7)
    up3 = UpSampling2D((2, 2))(conv6)
    drop8 = Dropout(args.dropout)(up3)
    decoded = Conv2D(config.patch_size[-1], (5, 5),
                     activation=None, padding='same',
                     kernel_regularizer=regularizers.l2(0.01))(drop8)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(loss='mean_squared_error',
                        optimizer=Adam(args.learning_rate))

    return autoencoder


def make_predictor(args, ae):

    from_ae = ae.get_layer('dense_2').output

    drop4 = Dropout(args.dropout)(from_ae)
    dense3 = Dense(128, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01))(drop4)
    drop5 = Dropout(args.dropout)(dense3)
    dense4 = Dense(64, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01))(drop5)
    drop6 = Dropout(args.dropout)(dense4)
    output = Dense(config.num_classes**onehot,
                   activation='softmax' if onehot else'sigmoid',
                   kernel_regularizer=regularizers.l2(0.01))(drop6)

    predictor = Model(inputs=ae.input, outputs=output)

    for i in range(len(predictor.layers) - 5):
        predictor.layers[i].trainable = False

    predictor.compile(
        loss='categorical_crossentropy' if onehot else'binary_crossentropy',
        optimizer=Adam(args.learning_rate),
        metrics=['accuracy'])
    return predictor


def train(model, data_model, h5s, args, n,
          callbacks, input_target, weights_path):
    X_train, X_test = train_test_split(
        h5s, test_size=0.2, random_state=config.random_state)
    X_train, X_val = train_test_split(
        X_train, test_size=0.2, random_state=config.random_state)

    if args.use_saved_weights and os.path.isfile(weights_path):
        print("loading weights")
        model.load_weights(weights_path)

    if args.train:
        X_train_batcher = data_model.as_batcher(
            X_train, config.batch_size, 9999999, input_target=input_target,
            wait_for_load=True)
        X_val_batcher = data_model.as_batcher(
            X_val, config.batch_size, 9999999, input_target=input_target,
            wait_for_load=True)

        model.fit_generator(
            X_train_batcher,
            steps_per_epoch=n * len(X_train) / args.batch_size,
            epochs=args.epochs, verbose=1,
            callbacks=callbacks,
            validation_data=X_val_batcher,
            validation_steps=n * len(X_val) / args.batch_size,
            class_weight=None,
            max_queue_size=n / args.batch_size, workers=1,
            use_multiprocessing=False, initial_epoch=0
        )

    X_test_batcher = data_model.as_batcher(
        X_test, config.batch_size, 9999999, input_target=input_target)

    results = model.evaluate_generator(
        X_test_batcher, steps=n * len(X_test) / args.batch_size,
        max_queue_size=config.max_queue_size, workers=1,
        use_multiprocessing=True,
    )

    return results


def train_encoder(model, args):
    ae_weights_path = weights_path % (
        "ae", args.normalize, args.median_time, augment, args.close_size)

    checkpoint = callbacks.ModelCheckpoint(
        ae_weights_path,
        monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + "ae" + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    callbacks_list = [tb, earlyStopping, checkpoint]

    data_model = Model_data(
        config.patch_size, bag_size=config.bag_size,
        preprocess=args.normalize,
        annotation_groupname="", remove_unlabeled=False,
        from_h5=True, one_hot=False, median_time=args.median_time,
        normalize_wieghtshare=False, augment=False, negative=0,
        samples=image_samples, ignore_annotations=True,
        prioritize_close_background=args.close_size)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    train(model, data_model, h5s, args, image_samples,
          callbacks_list, True, ae_weights_path)


def train_predictor(model, args, earlyStopping=earlyStopping):
    pred_weights_path = weights_path % (
        "pred", args.normalize, args.median_time, augment, args.close_size)

    checkpoint = callbacks.ModelCheckpoint(
        pred_weights_path,
        monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + "pred" + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    callbacks_list = [tb, earlyStopping, checkpoint]

    n = int(image_samples / 2 * 4 ** augment)

    data_model = Model_data(
        config.patch_size, bag_size=config.bag_size,
        preprocess=args.normalize,
        annotation_groupname=config.c_anno_groupname,
        from_h5=True, one_hot=onehot, median_time=args.median_time,
        normalize_wieghtshare=True, augment=augment, negative=0,
        samples=int(image_samples / 4),
        prioritize_close_background=args.close_size)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    test_res = train(
        model, data_model, h5s, args, n, callbacks_list,
        False, pred_weights_path)
    # Semisupervised, normalized, median filter, loss, accuracy
    # print("True,%s,%i %%,%f,%f" % (
    #     str(args.normalized), args.median_time, test_res[0], test_res[1]))
    config.print_to_result("True", str(args.normalize),
                           args.median_time, test_res[0], test_res[1])

    return data_model


if __name__ == '__main__':

    args = config.get_args("autoencoder")
    # print(args)

    autoencoder = make_autoencoder_model(args)

    autoencoder.summary()

    train_encoder(autoencoder, args)
    gc.collect()

    if not args.skip_pred:

        predictor = make_predictor(args, autoencoder)

        predictor.summary()

        args.learning_rate = args.learning_rate * 1000
        data_model = train_predictor(predictor, args)

        debug.test_model(predictor, data_model, args, "semi")
