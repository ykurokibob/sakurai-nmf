"""Utility for construct optimizers"""

import collections
import tensorflow as tf

Layer = collections.namedtuple('Layer', 'kernel, bias, output, activation, use_bias')
Layer.__new__.__defaults__ = len(Layer._fields) * (None,)


def get_train_ops(graph=None):
    graph = graph or tf.get_default_graph()
    vars_ = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return vars_


def get_name(x):
    if x is None:
        return None
    return x.name.split('/')[0]


def _get_activation(op, graph=None):
    graph = graph or tf.get_default_graph()
    activations = ['Relu']
    _activation = None
    for activation in activations:
        try:
            _activation = _activation or \
                          graph.get_operation_by_name('{}/{}'.format(op, activation))
        except KeyError:
            pass
        if _activation:
            return _activation
    return _activation


def factorize_v_bias(v: tf.Tensor):
    # For example combined matrix (785, 100)
    # will be (784, 100) and (1, 100)
    size = v.shape.as_list()[0] - 1
    bias = tf.identity(
        tf.squeeze(v[size:, :]))
    v = tf.identity(v[:size, :])
    return v, bias


def zip_layer(inputs: tf.Tensor, ops: list, graph=None):
    """
    Args:
        inputs: Inputs of network
        ops: List of layers collected by tf.get_collection

    Returns:
        List of layers zipped by weight(kernel) and bias.
    """
    layers = []
    # Use temp operation for the last layer doesn't use bias.
    ops.append(None)
    
    outputs = inputs
    while ops:
        train_op = ops.pop(0)
        if train_op is None:
            return layers
        train_op_name = get_name(train_op)
        activation = _get_activation(train_op_name, graph=graph)
        # Conform whether the layer have the bias or not.
        maybe_bias_op = ops[0]
        if train_op_name == get_name(maybe_bias_op):
            
            # Checking the kernel's shape correspond to bias.
            kernel_shape = train_op.get_shape()
            bias_shape = maybe_bias_op.get_shape()
            assert kernel_shape[1] == bias_shape[0], "There is some bugs," \
                                                     "kernel doesn't correspond to bias."
            use_bias = True
            # bias ops no need.
            ops.pop(0)
        else:
            maybe_bias_op = None
            use_bias = False
        
        layers.append(Layer(kernel=train_op,
                            bias=maybe_bias_op,
                            output=outputs,
                            activation=activation,
                            use_bias=use_bias,
                            ))
        
        # Calculate hidden outputs
        hidden = tf.matmul(outputs, train_op)
        if use_bias:
            hidden = tf.add(hidden, maybe_bias_op)
        hidden = tf.identity(hidden, name='hidden')
        
        outputs = hidden