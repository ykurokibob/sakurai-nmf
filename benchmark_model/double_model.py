"""64bit model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import numpy as np
import tensorflow as tf

batch_size = 100


def build_model():
    g = tf.Graph()
    inputs = tf.placeholder(tf.float64, (None, 784), name='inputs')
    labels = tf.placeholder(tf.float64, (None, 1), name='labels')
    x = tf.layers.dense(inputs, 100)
    x = tf.layers.dense(x, 1)
    losses = tf.losses.mean_squared_error(labels=labels, predictions=x)
    loss = tf.reduce_mean(losses)
    return inputs, labels, loss


def build_data():
    x = np.random.uniform(0., 1., size=(batch_size, 784)).astype(np.float64)
    y = np.random.uniform(0., 1., size=(batch_size, 1)).astype(np.float64)
    return x, y


def get_train_ops(graph: tf.Graph):
    return graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def main(_):
    inputs, labels, loss = build_model()
    x, y = build_data()
    graph = tf.get_default_graph()
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        loss = sess.run(loss, feed_dict={inputs: x, labels: y})
        print('loss {}'.format(loss))
        pprint(get_train_ops(graph))


if __name__ == '__main__':
    # tf.enable_eager_execution()
    tf.app.run()