import config
import tensorflow as tf
# import numpy as np
from tqdm import tqdm
from model_data import Model_data
from sklearn.model_selection import train_test_split

# Training Parameters
learning_rate = 0.00001
num_steps = 2000
bag_size = 1
batch_size = 32

# Network Parameters
num_classes = 3
epochs = 30000
dropout = 0.25  # Dropout, probability to drop a unit
patch_size = (17, 17, 5)
median_time = 2

# Log parameters
logs_path = '/tmp/tensorflow_logs/simple_cnn/'
run_name = "test"
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
                logits=logits_train, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    with tf.variable_scope("Accuracy"):
        labels_1d = tf.argmax(labels, axis=1)
        acc_op = tf.metrics.accuracy(labels_1d, pred_classes)

    # Create a summary to monitor cost tensor
    summary_train = tf.summary.scalar("loss", loss_op)
    # Create a summary to monitor accuracy tensor
    summary_test = tf.summary.scalar("accuracy", acc_op)

    return (loss_op, train_op, acc_op,
            pred_classes, features, labels, summary_train, summary_test)


if __name__ == '__main__':

    data_model = Model_data(
        patch_size, bag_size=bag_size,
        from_h5=True, median_time=median_time, one_hot=True,
        normalize_wieghtshare=True, augment=True)
    h5s = config.get_h5()

    X_train, X_test = train_test_split(
        h5s, test_size=0.33, random_state=config.random_state)

    X_train_batcher = data_model.as_batcher(X_train, batch_size)
    X_test_batcher = data_model.as_batcher(X_test, batch_size)

    accs = []

    with tf.Session() as sess:

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

        t = tqdm(range(epochs))
        for epoch in t:

            X_batch, y_batch = X_train_batcher.next_batch()
            feed_dict = {X: X_batch, y: y_batch}
            _, summa = sess.run(
                [train_op, summary_train], feed_dict)

            if epoch % 10 == 0 and epoch:
                train_writer.add_summary(summa, epoch)
                X_batch, y_batch = X_test_batcher.next_batch()
                feed_dict = {X: X_batch, y: y_batch}
                val_acc, summa = sess.run(
                    [acc_op, summary_test], feed_dict)
                train_writer.add_summary(summa, epoch)
                t.set_postfix(loss=val_acc)

        t.close()
