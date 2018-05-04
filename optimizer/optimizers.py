"""Optimize neural network with Non-negative Matrix Factorization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import matrix_factorization as mf
from optimizer import utility


def default_config():
    return locals()


class NMFOptimizer(object):
    """Optimize model like backpropagation."""
    
    def __init__(self, config, model):
        """Optimize model like backpropagation.
        Args:
            config: configuration for setting optimizer.
            model: Neural network model.
        """
        self._config = config
        self._model = model
        self._init()
    
    def _init(self):
        self._ops = utility.get_train_ops()
        self._layers = utility.zip_layer(self._model.inputs, ops=self._ops)
    
    def minimize(self):
        """Construct the control dependencies for calculating neural net optimized.
        
        Returns:
            tf.no_op.
            The import
        """
        a = self._model.labels
        updates = []
        # Reverse and remove first element.
        layers = self._layers[::-1]
        for i, layer in enumerate(layers):
            u = layer.output
            v = layer.kernel
            if layer.use_bias:
                v = tf.concat((v, layer.bias[None, ...]), axis=0)
            
            # Not use activation (ReLU)
            if not layer.activation:
                u, v = mf.semi_nmf(a=a, u=u, v=v,
                                   use_tf=True,
                                   use_bias=layer.use_bias,
                                   )
            # Use activation (ReLU)
            else:
                u, v = mf.nonlin_semi_nmf(a=a, u=u, v=v,
                                          use_tf=True,
                                          use_bias=layer.use_bias,
                                          )
            if layer.use_bias:
                v, bias = utility.factorize_v_bias(v)
                updates.append(layer.bias.assign(bias))
            updates.append(layer.kernel.assign(v))
            a = tf.identity(u)
        return tf.group(*updates)
