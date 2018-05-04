from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import tensorflow as tf

import benchmark_model
from optimizer import optimizers

batch_size = 100
label_size = 1


def default_config():
    return locals()


class NMFOptimizerTest(tf.test.TestCase):
    
    def test_factorize(self):
        print()
        model = benchmark_model.build_tf_model()
        config = default_config()
        optimizer = optimizers.NMFOptimizer(config, model)
        train_op = optimizer.minimize()
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
    
    def test_mnist(self):
        model = benchmark_model.build_tf_model()
        mnist = benchmark_model.mnist()
        x, y = mnist.get_batch(batch_size=batch_size)
        assert x.shape == (batch_size, 784)
        assert y.shape == (batch_size, 1)
        
        config = default_config()
        optimizer = optimizers.NMFOptimizer(config, model)
        train_op = optimizer.minimize()
        losses = []
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
            for i in range(5):
                x, y = mnist.get_batch(batch_size=batch_size)
                old_loss = sess.run(model.loss, feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
                _ = sess.run(train_op, feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
                new_loss = sess.run(model.loss, feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
                losses.append(new_loss)
                print('\nold loss {}, new loss {}'.format(old_loss, new_loss), end='', flush=True)
        
        # import matplotlib.pyplot as plt
        # plt.loglog(losses)
        # plt.show()