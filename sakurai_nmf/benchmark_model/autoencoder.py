"""Vanilla Autoencoder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import tensorflow as tf

from sakurai_nmf.benchmark_model import double_model

embedding_dim = 32


def build_tf_model(batch_size, shape=784, use_bias=False, activation=None):
    inputs = tf.placeholder(tf.float64, (batch_size, shape), name='inputs')
    
    encoded = tf.layers.Dense(embedding_dim)(inputs)
    
    decoded = tf.layers.Dense(784, activation=tf.nn.sigmoid)(encoded)
    
    losses = tf.losses.mean_squared_error(labels=inputs, predictions=decoded)
    loss = tf.reduce_mean(losses)
    
    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = optimizer.minimize(loss)
    return agents.tools.AttrDict(loss=loss,
                                 inputs=inputs,
                                 train_op=train_op)


def main(_):
    batch_size = 3000
    (x_train, y_train), (x_test, y_test) = double_model.load_one_hot_data()
    model = build_tf_model(batch_size)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            x, y = double_model.batch(x_train, y_train, batch_size)
            _, loss = sess.run([model.train_op, model.loss], feed_dict={model.inputs: x})
            print(loss)


if __name__ == '__main__':
    tf.app.run()