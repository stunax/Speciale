import config
import tensorflow as tf
from tqdm import tqdm
from model_data import Model_data
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
from mt_batcher import mt_batcher


# Training Parameters
learning_rate = 0.0000001
num_steps = 2000
bag_size = 2
batch_size = 128

# Network Parameters
num_classes = 3
epochs = 300000
val_freq = 100
dropout = 0.50  # Dropout, probability to drop a unit
patch_size = (17, 17, 5)
median_time = 2

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


def loss(logits, labels, target=""):

    with tf.name_scope("loss_%s" % target):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels))

# Define the model function (following TF Estimator Template)


def model_fn(train_features, train_labels, test_features, test_labels):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time,
    # we need to create 2 distinct computation graphs that still share the same
    # weights.
    logits_train = conv_net(train_features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(test_features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # Define loss and optimizer
    loss_op_train = loss(logits_train, train_labels, "train")
    loss_op_test = loss(logits_test, test_labels, "test")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op_train,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    with tf.variable_scope("Accuracy"):
        labels_1d = tf.argmax(test_labels, axis=1)
        acc_op, _ = tf.metrics.accuracy(labels_1d, pred_classes)

    # Create a summary to monitor cost tensor
    summary_train = tf.summary.scalar("loss_train", loss_op_train)
    # Create a summary to monitor accuracy tensor
    summary_test_loss = tf.summary.scalar("loss_test", loss_op_test)
    summary_test_acc = tf.summary.scalar("accuracy", acc_op)
    summary_test = tf.summary.merge([summary_test_acc, summary_test_loss])

    return (loss_op_train, train_op, summary_train,
            loss_op_test, acc_op, summary_test)


def end(train_batcher, test_batcher, sess):
    test_batcher.close(sess)
    train_batcher.close(sess)


def load_data():

    data_model = Model_data(
        patch_size, bag_size=bag_size,
        from_h5=True, one_hot=True, median_time=median_time,
        normalize_wieghtshare=True, augment=True)
    h5s = config.get_h5(
        annotation_name=config.c_anno_groupname, ignore_1_2=True)

    X_train, X_test = train_test_split(
        h5s, test_size=0.33, random_state=config.random_state)

    # Prime numbers used
    X_train_batcher = data_model.as_batcher(
        X_train, batch_size, batch_size * 307 * bag_size)
    X_test_batcher = data_model.as_batcher(
        X_test, batch_size, batch_size * 179 * bag_size)

    train_batcher = mt_batcher(patch_size, X_train_batcher, batch_size)
    test_batcher = mt_batcher(patch_size, X_test_batcher, batch_size)

    train_X, train_y = train_batcher.make_queue()
    test_X, test_y = test_batcher.make_queue()

    train_batcher.start_queuing()
    test_batcher.start_queuing()

    return train_X, train_y, test_X, test_y, [train_batcher, test_batcher]


if __name__ == '__main__':

    accs = []

    with tf.Session() as sess:

        train_X, train_y, test_X, test_y, batchers = load_data()

        # Build the Estimator
        loss_op_train, train_op, summary_train, \
            loss_op_test, acc_op, summary_test = model_fn(
                train_X, train_y, test_X, test_y)

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
            _, _, summa = sess.run(
                [loss_op_train, train_op, summary_train])

            if epoch % val_freq == 0:
                train_writer.add_summary(summa, epoch)
                loss_val, _, summa = sess.run(
                    [loss_op_test, acc_op, summary_test])
                test_writer.add_summary(summa, epoch)
                t.set_postfix(loss=loss_val)
        end(*batchers, sess)
        sess.close()
        t.close()
