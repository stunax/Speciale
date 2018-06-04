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

encoded_size = 128
patch_size = (32, 32, 5)
image_samples = 300000
onehot = True
augment = False

weights_path = config.weights_path


def make_autoencoder_model(args):

    input_img = Input(shape=patch_size)
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
    encoded = Dense(encoded_size, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(dense1)
    # decoder
    dense2 = Dense(1024,
                   kernel_regularizer=regularizers.l2(0.01))(encoded)
    reshape1 = Reshape((4, 4, 64))(dense2)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(reshape1)
    up1 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(64, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up1)
    up2 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(32, (5, 5), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(up2)
    up3 = UpSampling2D((2, 2))(conv6)
    decoded = Conv2D(patch_size[-1], (5, 5),
                     activation=None, padding='same',
                     kernel_regularizer=regularizers.l2(0.01))(up3)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(loss='mean_squared_error',
                        optimizer=Adam(args.learning_rate))

    return autoencoder


def make_predictor(args, ae):

    from_ae = ae.get_layer('dense_2').output

    drop4 = Dropout(args.dropout)(from_ae)
    dense3 = Dense(64, activation='relu')(drop4)
    drop5 = Dropout(args.dropout)(dense3)
    dense4 = Dense(32, activation='relu')(drop5)
    drop6 = Dropout(args.dropout)(dense4)
    output = Dense(config.num_classes**onehot,
                   activation='softmax' if onehot else'sigmoid')(drop6)

    predictor = Model(inputs=ae.input, outputs=output)

    for i in range(len(predictor.layers) - 5):
        predictor.layers[i].trainable = False

    predictor.compile(
        loss='categorical_crossentropy' if onehot else'binary_crossentropy',
        optimizer=Adam(args.learning_rate),
        metrics=['accuracy'])
    return predictor


def train(model, data_model, h5s, args, n, callbacks, input_target):
    X_train, X_test = train_test_split(
        h5s, test_size=0.2, random_state=config.random_state)
    X_train, X_val = train_test_split(
        X_train, test_size=0.2, random_state=config.random_state)

    X_train_batcher = data_model.as_batcher(
        X_train, config.batch_size, 9999999, input_target=input_target,
        wait_for_load=True)
    X_val_batcher = data_model.as_batcher(
        X_val, config.batch_size, 9999999, input_target=input_target,
        wait_for_load=True)

    model.fit_generator(
        X_train_batcher, steps_per_epoch=n * len(X_train) / args.batch_size,
        epochs=config.max_epochs, verbose=1,
        callbacks=callbacks,
        validation_data=X_val_batcher,
        validation_steps=n * len(X_val) / args.batch_size,
        class_weight=None,
        max_queue_size=n / args.batch_size, workers=1,
        use_multiprocessing=True, initial_epoch=0
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
        "ae", args.normalize, args.median_time, augment)

    if config.use_saved_weights and os.path.isfile(ae_weights_path):
        print("loading weights")
        model.load_weights(ae_weights_path)
        return

    checkpoint = callbacks.ModelCheckpoint(
        ae_weights_path,
        monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + "ae" + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    callbacks_list = [tb, lr_decay, earlyStopping, checkpoint]

    data_model = Model_data(
        patch_size, bag_size=1,
        preprocess=args.normalize,
        annotation_groupname="", remove_unlabeled=False,
        from_h5=True, one_hot=False, median_time=args.median_time,
        normalize_wieghtshare=False, augment=False, negative=0,
        samples=image_samples, ignore_annotations=True)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    train(model, data_model, h5s, args, image_samples, callbacks_list, True)


def train_predictor(model, args):
    pred_weights_path = weights_path % (
        "pred", args.normalize, args.median_time, augment)

    # if False:  # use_saved_weights and os.path.isfile(pred_weights_path):
    if config.use_saved_weights and os.path.isfile(pred_weights_path):
        print("loading weights")
        model.load_weights(pred_weights_path)
        return

    checkpoint = callbacks.ModelCheckpoint(
        pred_weights_path,
        monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
    tb = callbacks.TensorBoard(
        log_dir=config.logs_path + "pred" + args.run_name,
        batch_size=args.batch_size, histogram_freq=config.histogram_freq)
    callbacks_list = [tb, lr_decay, earlyStopping, checkpoint]

    n = 30000 * 4 ** augment

    data_model = Model_data(
        patch_size, bag_size=config.bag_size,
        preprocess=args.normalize,
        annotation_groupname=config.c_anno_groupname,
        prioritize_close_background=20,
        from_h5=True, one_hot=onehot, median_time=args.median_time,
        normalize_wieghtshare=True, augment=augment, negative=0)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    test_res = train(
        model, data_model, h5s, args, n, callbacks_list, False)
    # Semisupervised, normalized, median filter, loss, accuracy
    # print("True,%s,%i %%,%f,%f" % (
    #     str(args.normalized), args.median_time, test_res[0], test_res[1]))
    config.print_to_result("True", str(args.normalize),
                           args.median_time, test_res[0], test_res[1])


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Conv net test.")
    parser.add_argument('--epochs', default=config.max_epochs, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument(
        '--run_name', default='autoencoder/' + config.run_name)
    parser.add_argument('--learning_rate', default=config.learning_rate,
                        type=float, help="Initial learning rate")
    parser.add_argument('--normalize', default=config.normalize_input,
                        type=bool, help="normalize images?")
    parser.add_argument('--median_time', default=config.median_time, type=int,
                        help="Median time filter the data?")
    parser.add_argument('--dropout', default=config.dropout, type=int,
                        help="Dropout")
    args = parser.parse_args()
    # print(args)

    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (0.8 ** epoch))
    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='auto')

    autoencoder = make_autoencoder_model(args)

    # autoencoder.summary()

    train_encoder(autoencoder, args)
    gc.collect()

    predictor = make_predictor(args, autoencoder)

    # predictor.summary()

    args.learning_rate = args.learning_rate * 100
    train_predictor(predictor, args)
