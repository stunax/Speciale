from keras import backend as K
from keras import callbacks
from keras_autoencoder import make_autoencoder_model, make_predictor, \
    train_predictor, train_encoder
import config
import gc
import debug
import os.path

encoded_size = 128
# patch_size = (32, 32, 5)
image_samples = 30000
onehot = True
augment = True

weights_path = config.weights_path


def train_encoder2(model, args):
    ae_weights_path = weights_path % (
        "ae", args.normalize, args.median_time, augment, args.close_size)
    print(ae_weights_path)

    if os.path.isfile(ae_weights_path):
        model.load_weights(ae_weights_path)
    else:
        print("autoencoder not found. training autoencoder.")
        train_encoder(model, args)


def run(earlyStopping):
    autoencoder = make_autoencoder_model(args)

    train_encoder2(autoencoder, args)
    gc.collect()

    predictor = make_predictor(args, autoencoder)

    predictor.summary()

    args.learning_rate = args.learning_rate * 10000
    data_model = train_predictor(predictor, args, earlyStopping=earlyStopping)

    debug.test_model(predictor, data_model, args, "semi")

    K.clear_session()


if __name__ == '__main__':

    args = config.get_args("autoencoder")
    # print(args)

    earlyStopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='auto')
    run_name = args.run_name
    args.use_saved_weights = 0
    args.train = 1
    for i in range(1):
        args.run_name = run_name
        run(earlyStopping)
