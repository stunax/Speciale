import numpy as np
import config
import tensorflow as tf
from tqdm import tqdm, trange
from model_data import Model_data
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
from imageio import imwrite
from helpers import cat_image_to_gray


# Training Parameters
learning_rate = 0.00001
num_steps = 2000
bag_size = 4
batch_size = 128

# Network Parameters
num_classes = 2
epochs = 30000
max_true_epochs = 20
dropout = 0.50  # Dropout, probability to drop a unit
patch_size = (17, 17, 5)

# Preprocessing parameters
normalize_input = True
median_time = 0

# Log parameters
logs_path = '/tmp/tensorflow_logs/simple_cnn/'
run_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
# Create the neural network


def conv_net(X, n_classes, dropout, reuse, is_training, k):
    # x = tf.image.rot90(X, k=k)
    x = X
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # conv1 = tf.contrib.layers.batch_norm(
        #    conv1, scale=True, is_training=is_training)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # conv2 = tf.contrib.layers.batch_norm(
        #    conv2, scale=True, is_training=is_training)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn():

    features = tf.placeholder(tf.float32, shape=(None,) + patch_size, name="X")
    labels = tf.placeholder(tf.int32, shape=(None,), name="y")
    labels_onehot = tf.one_hot(labels, num_classes, dtype=tf.int32)
    k = tf.placeholder(tf.int32, name="k")
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time,
    # we need to create 2 distinct computation graphs that still share the same
    # weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True, k=k)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False, k=k)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # Define loss and optimizer
    with tf.variable_scope("loss"):
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits_train, labels=labels_onehot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    with tf.variable_scope("Accuracy"):
        labels_1d = tf.argmax(labels_onehot, axis=1)
        acc_op, _ = tf.metrics.accuracy(labels_1d, pred_classes)

    # Create a summary to monitor cost tensor
    summary_train = tf.summary.scalar("loss", loss_op)
    # Create a summary to monitor accuracy tensor
    summary_test = tf.summary.scalar("accuracy", acc_op)

    return (loss_op, train_op, acc_op,
            pred_classes, features, labels, summary_train, summary_test)


def dnn_pred_image(data, sess, op, X):
    preds = []
    for i in trange(int(data.shape[0] / batch_size)):
        from_i = i * batch_size
        to_i = min(data.shape[0], (i + 1) * batch_size)
        batch = data[from_i:to_i].reshape((-1,) + patch_size)
        preds.append(sess.run(op, feed_dict={X: batch}))
    return np.concatenate(preds, axis=0)


def pred_image(x):
    h5s, pred_classes, X, sess = x
    print(h5s)
    data, _ = data_model.handle_images([h5s])
    pred = dnn_pred_image(data, sess, pred_classes, X)
    pred = pred.reshape((1024, 1024))
    pred = cat_image_to_gray(pred)
    if np.sum(pred) == 0:
        pred = pred.astype(np.unit8)
    imwrite(config.test_imag_path + h5s[1] + "_pred.png", pred)
    data, _ = data_model2.handle_images([h5s])
    data = data.reshape((1024, 1024)).astype(np.uint8)
    imwrite(config.test_imag_path + h5s[1] + "_org.png", data)


def _get_near(t, z, tp, zp):
    return (str(t + tp).zfill(3), str(z + zp).zfill(2))


def get_near(h5s, pred_classes, X, sess):
    nears = []
    ps = [-1, 0, 1]
    for h5, df in h5s:
        t, z = map(int, df.split("_"))
        for tp in ps:
            for zp in ps:
                nears.append(((h5, "%s_%s" % _get_near(t, z, tp, zp)),
                              pred_classes, X, sess))
    return nears


if __name__ == '__main__':

    data_model = Model_data(
        patch_size, bag_size=bag_size, preprocess=normalize_input,
        annotation_groupname=config.c_anno_groupname,
        from_h5=True, one_hot=False, median_time=median_time,
        normalize_wieghtshare=True, augment=True, negative=0)
    data_model2 = Model_data(
        (1, 1, 1), bag_size=1,
        annotation_groupname="",
        from_h5=True, one_hot=False, median_time=0,
        normalize_wieghtshare=False, augment=False, remove_unlabeled=False)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    X_train, X_test = train_test_split(
        h5s, test_size=0.33, random_state=config.random_state)

    X_train_batcher = data_model.as_batcher(
        #    h5s, batch_size, 9999999999)
        X_train, batch_size, 9999999999)
    X_test_batcher = data_model.as_batcher(
        X_test, batch_size, 9999999999)

    accs = []

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:

        # Build the Estimator
        loss_op, train_op, acc_op, pred_classes, X, y,\
            summary_train, summary_test = model_fn()
        train_writer = tf.summary.FileWriter(
            logs_path + run_name + "train", graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(
            logs_path + run_name + "test")
        # * 4 because 4 times because of rotations
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        losses = []

        t = tqdm(range(epochs))
        for epoch in t:
            if X_train_batcher.epoch > max_true_epochs:
                break

            X_batch, y_batch = X_train_batcher.next_batch()
            feed_dict = {X: X_batch, y: y_batch}
            _, summa = sess.run(
                [train_op, summary_train], feed_dict)

            if epoch % 10 == 0:
                train_writer.add_summary(summa, epoch)
                X_batch, y_batch = X_test_batcher.next_batch()
                feed_dict = {X: X_batch, y: y_batch}
                loss_val, summa = sess.run(
                    [loss_op, summary_train], feed_dict)
                losses.append(loss_val)
                test_writer.add_summary(summa, epoch)
                postfix = np.mean(losses[-120:])
                t.set_postfix(loss=postfix)
                # if postfix < 0.22:
                #    break
        t.close()
        nears = get_near(X_train, pred_classes, X, sess)
        data_model.annotation_groupname = ""
        data_model.normalize_wieghtshare = False
        data_model.augment = False
        data_model.remove_unlabeled = False
        data_model.bag_size = 1
        for x in nears:
            pred_image(x)
        # map(pred_image, nears)
