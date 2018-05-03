"""64bit model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import numpy as np
import tensorflow as tf
from keras.datasets.mnist import load_data
from losses import frobenius_norm

batch_size = 500
label_size = 1


def build_tf_model():
    inputs = tf.placeholder(tf.float64, (batch_size, 784), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, label_size), name='labels')
    x = tf.layers.dense(inputs, 100, activation=tf.nn.relu, use_bias=False)
    x = tf.layers.dense(x, 50, use_bias=False, activation=tf.nn.relu)
    outputs = tf.layers.dense(x, label_size, activation=None, use_bias=False)
    # losses = tf.losses.mean_squared_error(labels=labels, predictions=outputs)
    losses =  frobenius_norm(labels, outputs)
    loss = tf.reduce_mean(losses)
    
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 loss=loss,
                                 )


def build_keras_model():
    inputs = tf.keras.Input((784,), batch_size=batch_size, dtype=tf.float64, name='inputs')
    labels = tf.keras.Input((label_size,), batch_size=batch_size, name='labels', dtype=tf.float64)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu, dtype=tf.float64)(inputs)
    x = tf.keras.layers.Dense(50, activation=tf.nn.relu, use_bias=False, dtype=tf.float64)(x)
    outputs = tf.keras.layers.Dense(label_size, dtype=tf.float64)(x)
    losses = tf.keras.losses.mean_squared_error(y_true=labels, y_pred=outputs)
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


def shuffle(x, y):
    assert len(x) == len(y)
    size = len(x)
    random = np.arange(0, size)
    np.random.shuffle(random)
    x = x[random]
    y = y[random]
    return x, y


class mnist():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
        x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
        y_train = y_train[..., None].astype(np.float64)
        self.x_train, self.y_train = shuffle(x_train, y_train)
    
    def get_batch(self, batch_size):
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        return self.x_train[:batch_size], self.y_train[:batch_size]


def get_train_ops(graph: tf.Graph):
    return graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def default_config():
    num_epochs = 100
    
    return locals()


if __name__ == '__main__':
    tf.app.run()