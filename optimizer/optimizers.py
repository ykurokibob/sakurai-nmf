"""Optimize neural network with Non-negative Matrix Factorization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matrix_factorization as mf
from optimizer import utility
import tensorflow as tf


def default_config():
    return locals()


class NMFOptimizer(object):
    
    def __init__(self, config, model):
        self._config = config
        self._model = model
        self._init()
        # self._factorize()
    
    def _init(self):
        self._ops = utility.get_train_ops()
        self._layers = utility.zip_layer(self._model.inputs, ops=self._ops)
    
    def minimize(self):
        a = self._model.labels
        update = []
        # Reverse and remove first element.
        layers = self._layers[::-1]
        for i, layer in enumerate(layers):
            u = layer.output
            v = layer.kernel
            print(a, u, v)
            if i == 0:
                u, v = mf.semi_nmf(a=a, u=u, v=v,
                                   use_tf=True,
                                   use_bias=False, # layer.use_bias
                                   )
            else:
                u, v = mf.nonlin_semi_nmf(a=a, u=u, v=v,
                                          use_tf=True,
                                          use_bias=False, # layer.use_bias
                                          )
            update.append(layer.kernel.assign(v))
            a = tf.identity(u)
        with tf.control_dependencies(update):
            return tf.no_op()
