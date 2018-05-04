"""Main of mnist model."""

import agents
import numpy as np
import tensorflow as tf

import benchmark_model
from optimizer import NMFOptimizer


def default_config():
    # Batch size
    batch_size = benchmark_model.batch_size
    # Number of matrix factorization epochs
    num_mf_epochs = 3
    # Number of back propagation epochs
    num_bp_epochs = 5
    # Learning rate for adam
    learning_rate = 0.01
    return locals()


def main(_):
    # Build one hot mnist model.
    model = benchmark_model.build_tf_one_hot_model()
    # Set configuration
    config = agents.tools.AttrDict(default_config())
    # Load one hot mnist data.
    (x_train, y_train), (x_test, y_test) = benchmark_model.load_one_hot_data()
    
    # Testing whether the dataset have correct shape.
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 10)
    
    # Minimize model's loss with NMF optimizer.
    optimizer = NMFOptimizer(config, model)
    train_op = optimizer.minimize()
    
    # Minimize model's loss with Adam optimizer.
    bp_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    bp_train_op = bp_optimizer.minimize(model.other_loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('NMF-optimizer')
        # Train with NMF optimizer.
        for i in range(config.num_mf_epochs):
            # Train...
            x, y = benchmark_model.batch(x_train, y_train, batch_size=config.batch_size)
            _, train_loss, train_acc = sess.run([train_op, model.other_loss, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            stats = []
            # Compute test accuracy.
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
        
        # Train with Adam optimizer.
        for i in range(config.num_bp_epochs):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=config.batch_size)
            _, train_loss, train_acc = sess.run([bp_train_op, model.other_loss, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            stats = []
            # Compute test accuracy.
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