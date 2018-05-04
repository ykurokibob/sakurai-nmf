"""Main of mnist model."""

import agents
import numpy as np
import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.utils.np_utils import to_categorical

import benchmark_model
from optimizer import NMFOptimizer


def default_config():
    batch_size = benchmark_model.batch_size
    num_mf_epochs = 2
    num_bp_epochs = 5
    learning_rate = 0.01
    return locals()


def main(_):
    model = benchmark_model.build_tf_one_hot_model()
    (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
    x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
    y_train = to_categorical(y_train, 10).astype(np.float64)
    x_test = x_test.reshape((-1, 784)).astype(np.float64) / 255.
    y_test = to_categorical(y_test, 10).astype(np.float64)
    
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 10)
    
    config = agents.tools.AttrDict(default_config())
    optimizer = NMFOptimizer(config, model)
    train_op = optimizer.minimize()
    
    bp_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    bp_train_op = bp_optimizer.minimize(model.other_loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('NMF-optimizer')
        for i in range(config.num_mf_epochs):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=config.batch_size)
            _, train_loss, train_acc = sess.run([train_op, model.other_loss, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            stats = []
            for _ in range(5):
                x, y = benchmark_model.batch(x_test, y_test, batch_size=config.batch_size)
                stats.append(sess.run([model.other_loss, model.accuracy], feed_dict={
                    model.inputs: x,
                    model.labels: y,
                }))
            test_loss, test_acc = np.mean(stats, axis=0)
            
            print('\r({}/{}) [Train]loss {}, accuracy {} [Test]loss {}, accuracy {}'.format(
                i + 1, config.num_mf_epochs,
                train_loss, train_acc, test_loss, test_acc), end='', flush=True)
        
        print("\n" + "=" * 10)
        print('Adam-optimizer')
        
        for i in range(config.num_bp_epochs):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=config.batch_size)
            _, train_loss, train_acc = sess.run([bp_train_op, model.other_loss, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            stats = []
            for _ in range(5):
                x, y = benchmark_model.batch(x_test, y_test, batch_size=config.batch_size)
                stats.append(sess.run([model.other_loss, model.accuracy], feed_dict={
                    model.inputs: x,
                    model.labels: y,
                }))
            test_loss, test_acc = np.mean(stats, axis=0)
            
            print('\r({}/{}) [Train]loss {}, accuracy {} [Test]loss {}, accuracy {}'.format(
                i + 1, config.num_bp_epochs,
                train_loss, train_acc, test_loss, test_acc), end='', flush=True)


if __name__ == '__main__':
    tf.app.run()