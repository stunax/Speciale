import config
import tensorflow as tf
import numpy as np
import tqdm
from model_data import Model_data
from sklearn.model_selection import train_test_split, KFold

# Training Parameters
learning_rate = 0.001
num_steps = 2000
bag_size = 10

# Network Parameters
num_classes = 2
epochs = 20
dropout = 0.25  # Dropout, probability to drop a unit
patch_size = (18, 18, 5)
median_time = 2

# Create the neural network


def conv_net(X, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # x = tf.reshape(X, shape=(None,) + patch_size)
        x = X

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

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
    labels = tf.placeholder(tf.int32, shape=(None, num_classes), name="y")
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time,
    # we need to create 2 distinct computation graphs that still share the same
    # weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # Define loss and optimizer
    with tf.variable_scope("loss"):
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits_train, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    print(labels.get_shape())
    print(pred_classes.get_shape())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    return loss_op, train_op, acc_op, pred_classes


if __name__ == '__main__':

    data_model = Model_data(patch_size, bag_size=bag_size,
                            from_h5=True, median_time=median_time)
    h5s = config.get_h5()

    X_train, X_test = train_test_split(
        h5s, test_size=0.33, random_state=config.random_state)

    # Build the Estimator
    loss_op, train_op, acc_op, pred_classes = model_fn()

    accs = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        pbar = tqdm(total=int(epochs * len(X_train) / bag_size) * 2)
        for i in range(epochs):
            accs_epoch = []
            for X_batch, y_batch in data_model.as_iter(X_train):
                feed_dict = {"X": X_batch, "y": y_batch}
                loss, _, acc = sess.run([loss_op, train_op, acc_op], feed_dict)
                accs_epoch.append(acc)
                pbar.update(1)
            accs.append(np.mean(accs_epoch))

            print("Training Accuracy: %f" % accs[-1])

            accs_epoch = []
            for X_batch, y_batch in data_model.as_iter(X_test):
                feed_dict = {"X": X_batch, "y": y_batch}
                pred_classes, acc = sess.run([pred_classes, acc_op], feed_dict)
                accs_epoch.append(acc)
                pbar.update(1)
        print("Training Accuracy: %f" % np.mean(accs_epoch))
