from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import numpy as np
import tensorflow as tf

import benchmark_model
from optimizer import optimizers

batch_size = benchmark_model.batch_size
label_size = benchmark_model.label_size


def default_config():
    return locals()


class NMFOptimizerTest(tf.test.TestCase):
    
    def test_factorize(self):
        print()
        model = benchmark_model.build_tf_one_hot_model()
        
        config = default_config()
        optimizer = optimizers.NMFOptimizer(config, model)
        train_op = optimizer.minimize()
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
    
    def test_mnist(self):
        model = benchmark_model.build_tf_one_hot_model()
        from keras.utils.np_utils import to_categorical
        from keras.datasets.mnist import load_data
        (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
        x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
        y_train = to_categorical(y_train, 10).astype(np.float64)
        
        assert x_train.shape == (60000, 784)
        assert y_train.shape == (60000, 10)
        
        config = default_config()
        optimizer = optimizers.NMFOptimizer(config, model)
        train_op = optimizer.minimize()
        losses = []
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
            for i in range(20):
                x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
                _, new_loss, acc = sess.run([train_op, model.other_loss, model.accuracy], feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
                losses.append(new_loss)
                print('\nloss {}, accuracy {}'.format(new_loss, acc), end='', flush=True)
        
        # import matplotlib.pyplot as plt
        # plt.loglog(losses)
        # plt.show()
        
