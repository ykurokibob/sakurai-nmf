"""Utility for construct optimizers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

Layer = collections.namedtuple('Layer', 'kernel, bias, recurrent, output, activation, use_bias')
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
        return ''
    return x.name.split('/')[0]


def get_op_name(x):
    if x is None:
        return ''
    return x.name.split('/')[-1]


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
        
        bias_op = None
        recurrent_op = None
        use_bias = False
        
        if 'recurrent_kernel' in get_op_name(ops[0]):
            # Checking the kernel's shape correspond to recurrent kernel.
            kernel_shape = train_op.get_shape()
            recurrent_kernel_shape = ops[0].get_shape()
            assert len(recurrent_kernel_shape) == 2, \
                "There is some bugs,kernel doesn't correspond to recurrent kernel."
            assert kernel_shape[1] == recurrent_kernel_shape[0], \
                "There is some bugs,kernel doesn't correspond to recurrent kernel."
            recurrent_op = ops[0]
            # bias ops no need.
            ops.pop(0)
        
        # Conform whether the layer have the bias or not.
        if 'bias' in get_op_name(ops[0]):
            # Checking the kernel's shape correspond to bias.
            kernel_shape = train_op.get_shape()
            bias_shape = ops[0].get_shape()
            assert len(bias_shape) == 1, "There is some bugs," \
                                         "kernel doesn't correspond to bias."
            assert kernel_shape[1] == bias_shape[0], "There is some bugs," \
                                                     "kernel doesn't correspond to bias."
            use_bias = True
            bias_op = ops[0]
            # bias ops no need.
            ops.pop(0)
        
        layers.append(Layer(kernel=train_op,
                            bias=bias_op,
                            recurrent=recurrent_op,
                            output=outputs,
                            activation=activation,
                            use_bias=use_bias,
                            ))
        
        # Calculate hidden outputs
        # TODO :have BUGS that cannot forward correct operations.
        hidden = tf.matmul(outputs, train_op)
        if use_bias:
            hidden = tf.add(hidden, bias_op)
        hidden = tf.identity(hidden, name='hidden')
        
        outputs = hidden


def zip_layer2(outputs: tf.Tensor):
    # WARNING no guarantee to get 2 placeholder
    """Collect placeholder from loss.
    
    Args:
        ops: Trainable variable operations
        loss: Loss of Neural network.

    Returns:
        inputs (tf.Tensor): inputs of neural network
        labels: (tf.Tensor): labels of neural network
    """
    queue = collections.deque([outputs])
    variable_names = []
    explored_inputs = {outputs}
    
    # We do a BFS on the dependency graph of the input function to find
    # the variables.
    while len(queue) != 0:
        # print(queue)
        tf_obj = queue.popleft()
        if tf_obj is None:
            continue
        # The object put into the queue is not necessarily an operation,
        # so we want the op attribute to get the operation underlying the
        # object. Only operations contain the inputs that we can explore.
        if hasattr(tf_obj, "op"):
            tf_obj = tf_obj.op
        for input_op in tf_obj.inputs:
            if input_op not in explored_inputs:
                queue.append(input_op)
                explored_inputs.add(input_op)
                print('inputs11', input_op.name)
        if "Variable" in tf_obj.node_def.op:
            variable_names.append(tf_obj.node_def.name)
            print('vari!', variable_names)
    variables = collections.OrderedDict()
    variable_list = [
        v for v in tf.global_variables()
        if v.op.node_def.name in variable_names
    ]
    for v in variable_list:
        variables[v.op.node_def.name] = v
    return variables


def test_get_rnn_outputs(outputs: tf.Tensor):
    # WARNING no guarantee to get 2 placeholder
    """Collect placeholder from loss.
    
    Args:
        ops: Trainable variable operations
        loss: Loss of Neural network.

    Returns:
        inputs (tf.Tensor): inputs of neural network
        labels: (tf.Tensor): labels of neural network
    """
    queue = collections.deque([outputs])
    variable_names = []
    explored_inputs = {outputs}
    
    # We do a BFS on the dependency graph of the input function to find
    # the variables.
    while len(queue) != 0:
        # print(queue)
        tf_obj = queue.popleft()
        if tf_obj is None:
            continue
        # The object put into the queue is not necessarily an operation,
        # so we want the op attribute to get the operation underlying the
        # object. Only operations contain the inputs that we can explore.
        if hasattr(tf_obj, "op"):
            tf_obj = tf_obj.op
        for input_op in tf_obj.inputs:
            if input_op not in explored_inputs:
                queue.append(input_op)
                explored_inputs.add(input_op)
                print('inputs11', input_op.name)
                if input_op.name == 'simple_rnn/TensorArrayReadV3:0':
                    return input_op
        if "Variable" in tf_obj.node_def.op:
            variable_names.append(tf_obj.node_def.name)
            print('vari!', variable_names)
    variables = collections.OrderedDict()
    variable_list = [
        v for v in tf.global_variables()
        if v.op.node_def.name in variable_names
    ]
    for v in variable_list:
        variables[v.op.node_def.name] = v
    return variables