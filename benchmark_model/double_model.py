"""64bit model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import numpy as np
import tensorflow as tf

batch_size = 100
label_size = 3


def build_model():
    inputs = tf.placeholder(tf.float64, (batch_size, 784), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, label_size), name='labels')
    x = tf.layers.dense(inputs, 100, activation=tf.nn.relu)
    x = tf.layers.dense(x, 50, use_bias=False, activation=tf.nn.relu)
    # x = tf.layers.dense(inputs, 100, activation=None)
    # x = tf.layers.dense(x, 50, use_bias=False, activation=None)
    outputs = tf.layers.dense(x, label_size, activation=None)
    losses = tf.losses.mean_squared_error(labels=labels, predictions=outputs)
    loss = tf.reduce_mean(losses)
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 loss=loss,
                                 )


def build_data():
    x = np.random.uniform(0., 1., size=(batch_size, 784)).astype(np.float64)
    y = np.random.uniform(-1., 1., size=(batch_size, label_size)).astype(np.float64)
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
        ops = get_train_ops(graph)


if __name__ == '__main__':
    tf.app.run()