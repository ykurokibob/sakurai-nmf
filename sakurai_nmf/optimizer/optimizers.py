"""Optimize neural network with Non-negative Matrix Factorization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sakurai.matrix_factorization as mf
from . import utility


class NMFOptimizer(object):
    """Optimize model like backpropagation."""
    
    def __init__(self, config=None, graph=None):
        """Optimize model like backpropagation.
        Args:
            config: configuration for setting optimizer.
            model: Neural network model.
        """
        self._config = config
        self._graph = graph
    
    def _init(self, loss):
        self._ops = utility.get_train_ops(graph=self._graph)
        self.inputs, self.labels = utility.get_placeholder_ops(loss)
        self._layers = utility.zip_layer(self.inputs, ops=self._ops, graph=self._graph)
    
    def minimize(self, loss=None):
        """Construct the control dependencies for calculating neural net optimized.
        
        Returns:
            tf.no_op.
            The import
        """
        self._init(loss)
        
        a = self.labels
        updates = []
        # Reverse
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
                                   num_iters=1,
                                   )
            # Use activation (ReLU)
            else:
                u, v = mf.nonlin_semi_nmf(a=a, u=u, v=v,
                                          use_tf=True,
                                          use_bias=layer.use_bias,
                                          num_calc_v=1,
                                          num_calc_u=1,
                                          )
            if layer.use_bias:
                v, bias = utility.split_v_bias(v)
                updates.append(layer.bias.assign(bias))
            updates.append(layer.kernel.assign(v))
            a = tf.identity(u)
        
        return tf.group(*updates)