from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import agents
import numpy as np
import tensorflow as tf

from sakurai_nmf import benchmark_model
from sakurai_nmf.optimizer import optimizers


def default_config():
    batch_size = 3000
    label_size = 10
    use_bias = False
    use_autoencoder = True
    return locals()


class NMFOptimizerTest(tf.test.TestCase):
    
    def test_factorize(self):
        config = agents.tools.AttrDict(default_config())
        model = benchmark_model.build_tf_one_hot_model(config.batch_size)
        
        optimizer = optimizers.NMFOptimizer(config)
        train_op = optimizer.minimize(model.frob_norm)
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
    
    def test_autoencoder(self):
        print()
        config = agents.tools.AttrDict(default_config())
        model = benchmark_model.build_tf_one_hot_model(batch_size=config.batch_size)
        
        optimizer = optimizers.NMFOptimizer(config)
        train_op = optimizer.minimize(model.frob_norm)
        self.assertEqual(model.inputs.shape, optimizer.decoder.shape)

    
    def test_mnist(self):
        config = agents.tools.AttrDict(default_config())
        model = benchmark_model.build_tf_one_hot_model(config.batch_size)
        from keras.utils.np_utils import to_categorical
        from keras.datasets.mnist import load_data
        (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
        x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
        y_train = to_categorical(y_train, 10).astype(np.float64)
        
        assert x_train.shape == (60000, 784)
        assert y_train.shape == (60000, 10)
        
        optimizer = optimizers.NMFOptimizer()
        train_op = optimizer.minimize(model.frob_norm)
        losses = []
        
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            pprint(optimizer._layers)
            for i in range(3):
                x, y = benchmark_model.batch(x_train, y_train, config.batch_size)
                _, new_loss, acc = sess.run([train_op, model.cross_entropy, model.accuracy], feed_dict={
                    model.inputs: x,
                    model.labels: y,
                })
                losses.append(new_loss)
                print('\nloss {}, accuracy {}'.format(new_loss, acc), end='', flush=True)


if __name__ == '__main__':
    tf.test.main()