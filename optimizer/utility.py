"""Utility for construct optimizers"""

import collections
import tensorflow as tf

Layer = collections.namedtuple('Layer', 'kernel, bias, use_bias')
Layer.__new__.__defaults__ = len(Layer._fields) * (None,)


def get_train_ops(graph=None):
    graph = graph or tf.get_default_graph()
    vars_ = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return vars_


def get_name(x):
    if x is None:
        return None
    return x.name.split('/')[0]


def zip_layer(ops: list):
    """
    Args:
        ops: List of layers collected by tf.get_collection

    Returns:
        List of layers zipped by weight(kernel) and bias.
    """
    layers = []
    # use temp operation for the last layer doesn't use bias.
    ops.append(None)
    
    while ops:
        train_op = ops.pop(0)
        if train_op is None:
            return layers
        train_op_name = get_name(train_op)
        # Conform whether the layer have the bias or not.
        maybe_bias_op = ops[0]
        if train_op_name == get_name(maybe_bias_op):
            
            # Checking the kernel's shape correspond to bias.
            kernel_shape = train_op.get_shape()
            bias_shape = maybe_bias_op.get_shape()
            assert kernel_shape[1] == bias_shape[0], "There is some bugs," \
                                                     "kernel doesn't correspond to bias."
            layers.append(Layer(kernel=train_op,
                                bias=maybe_bias_op,
                                use_bias=True))
            # bias ops no need.
            ops.pop(0)
        else:
            layers.append(Layer(kernel=train_op,
                                use_bias=False))