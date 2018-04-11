import threading
import tensorflow as tf


class mt_batcher(object):
    """docstring for mt_batcher"""

    def __init__(self, patch_size, batcher, batch_size):
        super(mt_batcher, self).__init__()
        self.patch_size = patch_size
        self.batcher = batcher
        self.batch_size = batch_size

    def get_entry(self):
        batchX, batchy = self.batcher.next_batch()
        return batchX[0], batchy[0]

    def make_queue(self):
        # Return input and target tensor
        with tf.variable_scope('Queue'):
            self.queue_input = tf.placeholder(
                tf.float32, shape=self.patch_size)
            self.queue_labels = tf.placeholder(tf.int32, shape=(3,))
            self.queue = tf.FIFOQueue(
                capacity=3000000, dtypes=[
                    tf.float32, tf.int32], shapes=[self.patch_size, (3,)])
            self.enqueue_op = self.queuequeue.enque_many(
                [self.queue_input, self.queue_labels])
            self.close_op = self.queuequeue.close()
            dequeue_op = self.queuequeue.dequeue()
            self.batch = tf.train.shuffle_batch(
                [dequeue_op], batch_size=self.batch_size,
                capacity=3000000, min_after_dequeue=1400000)
            return self.batch

    def thread_func(self, sess):
        while True:
            batchX, batchy = self.get_entry()
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
