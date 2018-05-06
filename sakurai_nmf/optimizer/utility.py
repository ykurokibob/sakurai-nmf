"""Utility for construct optimizers"""

import collections
import numpy as np
import tensorflow as tf

Layer = collections.namedtuple('Layer', 'kernel, bias, output, activation, use_bias')
Layer.__new__.__defaults__ = len(Layer._fields) * (None,)


def get_train_ops(graph=None):
    graph = graph or tf.get_default_graph()
    vars_ = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return vars_


def get_placeholder_ops(loss: tf.Operation):
    # WARNING no guarantee to get 2 placeholder
    """Collect placeholder from loss.
    
    Args:
        ops: Trainable variable operations
        loss: Loss of Neural network.

    Returns:
        inputs (tf.Tensor): inputs of neural network
        labels: (tf.Tensor): labels of neural network
    """
    queue = collections.deque([loss])
    explored_inputs = {loss}
    
    labels = tf.no_op()
    inputs = tf.no_op()
    
    # We do a BFS on the dependency graph of the input function to find
    # the variables.
    _tf_obj = None
    while len(queue) != 0:
        tf_obj = queue.popleft()
        if tf_obj is None:
            continue
        if hasattr(tf_obj, 'op'):
            _tf_obj = tf_obj
            tf_obj = tf_obj.op
        if tf_obj.type == 'Placeholder':
            # TODO: we must implement more safely.
            if isinstance(labels, tf.Tensor):
                inputs = _tf_obj
            elif not isinstance(labels, tf.Tensor):
                labels = _tf_obj
        for input_op in tf_obj.inputs:
            if input_op not in explored_inputs:
                queue.append(input_op)
                explored_inputs.add(input_op)
    return inputs, labels


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


def split_v_bias(v: tf.Tensor):
    # For example combined matrix (785, 100)
    # will be (784, 100) and (1, 100)
    if isinstance(v, tf.Tensor):
        size = v.shape.as_list()[0] - 1
        bias = tf.identity(
            tf.squeeze(v[size:, :]))
        v = tf.identity(v[:size, :])
        return v, bias
    if isinstance(v, np.ndarray):
        size = v.shape[0] - 1
        bias = np.squeeze(v[size:, :])
        v = v[:size, :]
        return v, bias
    raise NotImplementedError('Not support type {}'.format(type(v)))


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