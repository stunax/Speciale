import numpy as np
import config
import tensorflow as tf
from tqdm import tqdm
from model_data import Model_data
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
from dnn import *  # model_fn, conv_net1, conv_net2
from debug import pred_image, get_near


# Network Parameters
epochs = 80000
# Log parameters
logs_path = config.logs_path + '/simple_cnn/'
run_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
# Create the neural network


if __name__ == '__main__':

    data_model = Model_data(
        config.patch_size, bag_size=config.bag_size,
        preprocess=config.normalize_input,
        annotation_groupname=config.c_anno_groupname,
        from_h5=True, one_hot=False, median_time=config.median_time,
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
        X_train, config.batch_size, 9999999999)
    X_test_batcher = data_model.as_batcher(
        X_test, config.batch_size, 9999999999)

    accs = []

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:

        # Build the Estimator
        loss_op_train, train_op, loss_op_test, pred_classes, X, y,\
            summary_train, summary_test = model_fn(
                config.patch_size, config.num_classes, conv_net1,
                config.dropout, config.learning_rate)
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
            if X_train_batcher.epoch > config.max_epochs:
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
                    [loss_op_test, summary_test], feed_dict)
                losses.append(loss_val)
                test_writer.add_summary(summa, epoch)
                postfix = np.mean(losses[-120:])
                t.set_postfix(loss=postfix)
                # if postfix < 0.22:
                #    break
        t.close()
        nears = get_near(X_train, pred_classes, X,
                         sess, data_model, data_model2)
        data_model.annotation_groupname = ""
        data_model.normalize_wieghtshare = False
        data_model.augment = False
        data_model.remove_unlabeled = False
        data_model.bag_size = 1
        for x in nears:
            pred_image(x)
        # map(pred_image, nears)
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
