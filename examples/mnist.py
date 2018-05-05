"""Main of mnist model."""

import agents
import functools
import numpy as np
import tensorflow as tf

import benchmark_model
from optimizer import NMFOptimizer


def default_config():
    # Batch size
    batch_size = benchmark_model.batch_size
    # Number of matrix factorization epochs
    num_mf_epochs = 2
    # Number of back propagation epochs
    num_bp_epochs = 5
    # Learning rate for adam
    learning_rate = 0.01
    # NMF actiovation
    activation = None
    # NMF use bias
    use_bias = True
    return locals()


def train_and_test(train_op, num_epochs, sess, model, x_train, y_train, x_test, y_test, batch_size=1):
    for i in range(num_epochs):
        # Train...
        x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
        _, train_loss, train_acc = sess.run([train_op, model.other_loss, model.accuracy], feed_dict={
            model.inputs: x,
            model.labels: y,
        })
        stats = []
        # Compute test accuracy.
        for _ in range(5):
            x, y = benchmark_model.batch(x_test, y_test, batch_size=batch_size)
            stats.append(sess.run([model.other_loss, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        test_loss, test_acc = np.mean(stats, axis=0)
        
        print('\r({}/{}) [Train]loss {}, accuracy {} [Test]loss {}, accuracy {}'.format(
            i + 1, num_epochs,
            train_loss, train_acc, test_loss, test_acc), end='', flush=True)
    print()


def main(_):
    # Set configuration
    config = agents.tools.AttrDict(default_config())
    # Build one hot mnist model.
    model = benchmark_model.build_tf_one_hot_model(use_bias=config.use_bias, activation=config.activation)
    # Load one hot mnist data.
    (x_train, y_train), (x_test, y_test) = benchmark_model.load_one_hot_data()
    
    # Testing whether the dataset have correct shape.
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 10)
    
    # Minimize model's loss with NMF optimizer.
    # optimizer = NMFOptimizer(config)
    optimizer = NMFOptimizer()
    train_op = optimizer.minimize(model.loss)
    
    # Minimize model's loss with Adam optimizer.
    bp_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    bp_train_op = bp_optimizer.minimize(model.other_loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        _train_and_test = functools.partial(train_and_test,
                                            sess=sess, model=model,
                                            x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            batch_size=config.batch_size)
        print('NMF-optimizer')
        # Train with NMF optimizer.
        _train_and_test(train_op, num_epochs=config.num_mf_epochs)
        
        print('Adam-optimizer')
        # Train with Adam optimizer.
        _train_and_test(bp_train_op, num_epochs=config.num_bp_epochs)


if __name__ == '__main__':
    tf.app.run()