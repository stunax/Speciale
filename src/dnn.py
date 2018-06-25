import tensorflow as tf


def conv_net2(X, n_classes, dropout, reuse, is_training, k):
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


def conv_net1(X, n_classes, dropout, reuse, is_training, k):
    # x = tf.image.rot90(X, k=k)
    x = X
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 42 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 42, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # conv1 = tf.contrib.layers.batch_norm(
        #    conv1, scale=True, is_training=is_training)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


def cross_entropy_soft(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))


# Define the model function (following TF Estimator Template)
def model_fn(patch_size, num_classes, conv_model, dropout, learning_rate):

    features = tf.placeholder(tf.float32, shape=(None,) + patch_size, name="X")
    labels = tf.placeholder(tf.int32, shape=(None, ), name="y")
    labels_onehot = tf.one_hot(labels, num_classes, dtype=tf.int32, axis=-1)
    print(labels_onehot.get_shape())
    k = tf.placeholder(tf.int32, name="k")
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time,
    # we need to create 2 distinct computation graphs that still share the same
    # weights.
    logits_train = conv_model(features, num_classes, dropout, reuse=False,
                              is_training=True, k=k)
    logits_test = conv_model(features, num_classes, dropout, reuse=True,
                             is_training=False, k=k)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # Define loss and optimizer
    with tf.variable_scope("loss"):
        loss_op_train = cross_entropy_soft(logits_train, labels_onehot)
        loss_op_test = cross_entropy_soft(logits_train, labels_onehot)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op_train,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    # with tf.variable_scope("Accuracy"):
    #     labels_1d = tf.argmax(labels_onehot, axis=1)
    #     acc_op, _ = tf.metrics.accuracy(labels_1d, pred_classes)

    # Create a summary to monitor cost tensor
    summary_train = tf.summary.scalar("loss train", loss_op_train)
    # Create a summary to monitor accuracy tensor
    summary_test = tf.summary.scalar("loss val", loss_op_test)

    return (loss_op_train, train_op, loss_op_test,
            pred_classes, features, labels, summary_train, summary_test)
