"""64bit RNN model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from sakurai_nmf.losses import frobenius_norm


def build_rnn_mnist(batch_size, use_bias=False, activation=None):
    time_steps = 28
    num_features = 28
    inputs = tf.keras.layers.Input((time_steps, num_features), dtype=tf.float64, name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, 10), name='labels')
    
    activation = None or activation
    x = tf.keras.layers.SimpleRNN(100, use_bias=use_bias, activation=activation)(inputs)
    outputs = tf.layers.dense(x, 10, activation=None, use_bias=use_bias)
    
    losses = frobenius_norm(labels, outputs)
    frob_norm = tf.reduce_mean(losses)
    other_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs)
    cross_entropy = tf.reduce_mean(other_losses)
    
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.
    
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 frob_norm=frob_norm,
                                 cross_entropy=cross_entropy,
                                 accuracy=accuracy,
                                 )


def load_mnist(dataset='mnist'):
    from keras.datasets.mnist import load_data
    if dataset == 'fashion':
        from keras.datasets.fashion_mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.astype(np.float64) / 255.
    y_train = to_categorical(y_train, 10).astype(np.float64)
    x_test = x_test.astype(np.float64) / 255.
    y_test = to_categorical(y_test, 10).astype(np.float64)
    return (x_train, y_train), (x_test, y_test)


def batch(x, y, batch_size):
    rand_index = np.random.choice(len(x), batch_size)
    return x[rand_index], y[rand_index]


def main(_):
    batch_size = 100
    epoch_size = 3
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_rnn_mnist(batch_size=batch_size, use_bias=True, activation=tf.nn.relu)
    
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(model.cross_entropy)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(epoch_size):
            for _ in range(len(x_train) // batch_size):
                x, y = batch(x_train, y_train, batch_size)
                _, = sess.run([train_op],
                              feed_dict={model.inputs: x, model.labels: y})
            x, y = batch(x_test, y_test, batch_size)
            loss, acc = sess.run([model.cross_entropy, model.accuracy],
                                 feed_dict={model.inputs: x, model.labels: y})
            print('({}/{}) loss {}, accuracy {}'.format(epoch + 1, epoch_size, loss, acc))


if __name__ == '__main__':
    tf.app.run()