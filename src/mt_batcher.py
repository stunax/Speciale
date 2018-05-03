import threading
import tensorflow as tf


class mt_batcher(object):
    """docstring for mt_batcher"""

    def __init__(self, patch_size, batcher, batch_size, capacity=1000000):
        super(mt_batcher, self).__init__()
        self.patch_size = patch_size
        self.batcher = batcher
        self.batch_size = batch_size
        self.capacity = capacity

    def get_entry(self):
        batchX, batchy = self.batcher.next_batch()
        return batchX, batchy

    def make_queue(self):
        # Return input and target tensor
        with tf.variable_scope('Queue'):
            # Input op
            self.queue_input = tf.placeholder(
                tf.float32, (self.batch_size,) + self.patch_size)
            # Label op
            self.queue_labels = tf.placeholder(
                tf.int32, shape=(self.batch_size,))
            # Labels aer label encoded, but not onehot encoded yet.
            self.queue_labels_one_hot = tf.one_hot(
                self.queue_labels, 3, dtype=tf.int32)
            # Make queue. Large capacity, to enable several images.
            self.queue = tf.FIFOQueue(
                capacity=self.capacity, dtypes=[tf.float32, tf.int32],
                shapes=[self.patch_size, (3)])
            # Create enqueue operation for data feeder
            self.enqueue_op = self.queue.enqueue_many(
                [self.queue_input, self.queue_labels_one_hot])
            self.close_op = self.queue.close()
            # Create batch, which contains
            dequeue_op = self.queue.dequeue()
            self.batch = tf.train.shuffle_batch(
                dequeue_op, batch_size=self.batch_size,
                capacity=self.capacity,
                min_after_dequeue=int(self.capacity / 2))
            return self.batch

    def get_epoch(self):
        return self.batcher.epoch

    def thread_func(self, sess):
        while True:
            batchX, batchy = self.get_entry()
            batchy[batchy == -1] = 2
            # if batchy == -1:
            #     batchy = 2
            sess.run(self.enqueue_op, feed_dict={
                     self.queue_input: batchX, self.queue_labels: batchy})

    def start_queuing(self, sess):
        enqueue_thread = threading.Thread(
            target=self.thread_func, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            coord=self.coord, sess=sess)

        return self.coord, self.threads

    def close(self, sess):
        sess.run(self.queue.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(self.threads)
