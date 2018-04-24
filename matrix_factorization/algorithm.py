from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from pinv import tf_pinv


class MatrixFactorize(object):
    
    def __init__(self, target_y, v_shape, u_shape, name, activation=None, rcond=1e-3, inputs=None):
        self.target_y = target_y
        self.v_shape = v_shape
        self.u_shape = u_shape
        self.activation = activation or tf.nn.relu
        self.rcond = rcond
        self.name = name
        self.inputs = inputs
    
    def compute_u(self):
        raise NotImplementedError
    
    def compute_v(self, u_op):
        raise NotImplementedError
    
    def _initialize_matrix(self, inputs, v_shape, u_shape):
        self.v = inputs or K.variable(tf.random_uniform(v_shape, minval=0, maxval=1), name="v")
        self.u = K.variable(tf.random_uniform(u_shape, minval=-1, maxval=1), name="u")
        self.logits = tf.matmul(self.v, self.u)
        # print(self.v)
        # print(self.u)
    
    def _use_bias(self):
        self.v_bias = tf.concat([self.v, tf.ones((self.v_shape[0], 1))], axis=1)
        self.bias = K.variable(tf.random_uniform((1, self.u_shape[1]), maxval=1))
        self.u_bias = tf.concat([self.u, self.bias], axis=0)
        self.logits = tf.matmul(self.v_bias, self.u_bias)
        assert K.get_variable_shape(self.v_bias)[1] == K.get_variable_shape(self.u_bias)[0], "cannot matmul _U and _V."
    
    def matrix_loss(self, metric):
        return metric(self.target_y, self.activation(self.logits))



class _SemiNMF(MatrixFactorize):
    
    def __init__(self, target_y, v_shape, u_shape, name, **kwargs):
        super(_SemiNMF, self).__init__(target_y, v_shape, u_shape, name, **kwargs)
        with tf.name_scope(self.name + "/initialize_matrix"):
            self._initialize_matrix(self.inputs, v_shape, u_shape)
    
    def compute_u(self):
        v_inv = tf_pinv(self.v, rcond=self.rcond)
        u_op = tf.assign(self.u, tf.matmul(v_inv, self.target_y))
        return u_op, tf.no_op()
    
    def compute_v(self, u_op):
        au = tf.matmul(self.target_y, self.u, transpose_b=True)
        au_abs = tf.abs(au)
        uu = tf.matmul(self.u, self.u, transpose_b=True)
        uu_abs = tf.abs(uu)
        
        ua_plus = (au_abs + au) / 2
        ua_minus = (au_abs - au) / 2
        
        uu_plus = (uu_abs + uu) / 2
        uu_minus = (uu_abs - uu) / 2
        
        sqrt = tf.sqrt(tf.divide(
            ua_plus + tf.matmul(self.v, uu_minus),
            ua_minus + tf.matmul(self.v, uu_plus)))
        v_op = tf.assign(self.v, self.v * sqrt)
        return v_op


class _BiasSemiNMF(_SemiNMF):
    
    def __init__(self, target_y, v_shape, u_shape, name,
                 alpha=1e-3,
                 beta=1e-5,
                 **kwargs):
        super(_BiasSemiNMF, self).__init__(target_y=target_y,
                                           v_shape=v_shape,
                                           u_shape=u_shape,
                                           name=name,
                                           **kwargs)
        with tf.name_scope(self.name + "/initialize_matrix"):
            self._initialize_matrix(self.inputs, v_shape, u_shape)
            self._use_bias()
            self.alpha = alpha
            self.beta = beta
        
        # print("logits", self.logits)
    
    def compute_u(self):
        vv_inv = tf.multiply(
            self.alpha,
            tf_pinv(tf.matmul(self.v_bias, self.v_bias, transpose_a=True), rcond=self.rcond)
        )
        vv_shape = K.get_variable_shape(vv_inv)
        Jv = tf.add(tf.eye(*vv_shape), vv_inv)
        
        v_inv = tf_pinv(self.v_bias, rcond=self.rcond)
        # https://www.tensorflow.org/api_docs/python/tf/matrix_solve_ls
        solver = tf.linalg.lstsq(
            Jv, tf.matmul(v_inv, self.target_y), l2_regularizer=self.alpha)
        u_op = tf.assign(self.u, tf.slice(solver, [0, 0], self.u_shape))
        bias_op = tf.assign(
            self.bias, tf.slice(input_=solver,
                                begin=[self.u_shape[0], 0],
                                size=[1, self.u_shape[1]])
        )
        return u_op, bias_op
    
    def compute_v(self, u_op):
        au = tf.matmul(self.target_y, self.u, transpose_b=True)
        au_abs = tf.abs(au)
        uu = tf.matmul(self.u_bias, self.u, transpose_b=True)
        uu_abs = tf.abs(uu)
        
        ua_plus = tf.add(au_abs, au) * 0.5
        ua_minus = tf.subtract(au_abs, au) * 0.5
        
        uu_plus = tf.add(uu_abs, uu) * 0.5
        uu_minus = tf.subtract(uu_abs, uu) * 0.5
        
        lambda_v = tf.multiply(self.beta, self.v)
        sqrt = tf.sqrt(tf.divide(
            ua_plus + tf.matmul(self.v_bias, uu_minus) + lambda_v,
            ua_minus + tf.matmul(self.v_bias, uu_plus) + lambda_v))
        v_op = tf.assign(self.v, tf.multiply(self.v, sqrt))
        return v_op
    
    def matrix_loss(self, metric):
        return metric(self.target_y, self.activation(self.logits))


class _NonlinearSemiNMF(MatrixFactorize):
    
    def __init__(self, target_y, v_shape, u_shape, name, **kwargs):
        super(_NonlinearSemiNMF, self).__init__(target_y, v_shape, u_shape, name, **kwargs)
        with tf.name_scope(self.name + "/initialize_matrix"):
            self._initialize_matrix(self.inputs, v_shape, u_shape)
            self.y_sub_vu = tf.subtract(self.target_y, self.activation(self.logits))  # for reuse at compute_v
    
    def compute_u(self):
        """
        :return: U operation, and bias op(tf.no_op)
        """
        v_inv = tf_pinv(self.v, rcond=self.rcond)
        v_y = tf.matmul(v_inv, self.y_sub_vu)
        u_op = tf.assign_add(self.u, v_y)
        return u_op, tf.no_op
    
    def compute_v(self, u_op):
        u_inv = tf_pinv(self.u, rcond=self.rcond)
        vu = tf.matmul(self.v, self.u)
        y_sub_vu = tf.subtract(self.target_y, self.activation(vu))
        # TODO: WHY CANNOT USE self.y_sub_vu
        y_u = tf.matmul(y_sub_vu, u_inv)
        v_op = tf.assign(self.v, self.activation(tf.add(self.v, y_u)))
        return v_op


class _BiasNonlinearSNMF(_NonlinearSemiNMF):
    
    def __init__(self, target_y, v_shape, u_shape, name,
                 alpha=1e-3,
                 beta=1e-5,
                 **kwargs):
        super(_BiasNonlinearSNMF, self).__init__(target_y=target_y,
                                                 v_shape=v_shape,
                                                 u_shape=u_shape,
                                                 name=name,
                                                 **kwargs)
        with tf.name_scope(self.name + "/initialize_matrix"):
            self._initialize_matrix(self.inputs, v_shape, u_shape)
            self._use_bias()
            self.alpha = alpha
            self.beta = beta
            
            # print("logits", self.logits)
            self.y_sub_vu = tf.subtract(self.target_y, self.activation(self.logits))  # for reuse at compute_v
    
    def compute_u(self):
        vv_inv = tf.multiply(
            self.alpha,
            tf_pinv(tf.matmul(self.v_bias, self.v_bias, transpose_a=True), rcond=self.rcond)
        )
        vv_shape = K.get_variable_shape(vv_inv)
        assert vv_shape[0] == vv_shape[1]
        Jv = tf.add(tf.eye(*vv_shape), vv_inv)
        
        v_inv = tf_pinv(self.v_bias, rcond=self.rcond)
        # https://www.tensorflow.org/api_docs/python/tf/matrix_solve_ls
        solver = tf.linalg.lstsq(
            Jv, tf.add(self.u_bias, tf.matmul(v_inv, self.y_sub_vu)), l2_regularizer=self.alpha)
        u_op = tf.assign(self.u, tf.slice(solver, [0, 0], self.u_shape))
        bias_op = tf.assign(
            self.bias, tf.slice(input_=solver,
                                begin=[self.u_shape[0], 0],
                                size=[1, self.u_shape[1]])
        )
        return u_op, bias_op
    
    def compute_v(self, u_op):
        uu_inv = tf.multiply(
            self.beta,
            tf_pinv(tf.matmul(self.u, self.u, transpose_b=True), rcond=self.rcond)
        )
        uu_shape = K.get_variable_shape(uu_inv)
        assert uu_shape[0] == uu_shape[1]
        Ju = tf.add(tf.eye(*uu_shape), uu_inv)
        
        u_inv = tf_pinv(self.u, rcond=self.rcond)
        y_sub_f = tf.subtract(self.target_y, self.activation(tf.matmul(self.v_bias, self.u_bias)))
        u_add_yu = tf.add(self.v, tf.matmul(y_sub_f, u_inv))
        solver = tf.assign(self.v, tf.matmul(u_add_yu, tf.linalg.inv(Ju)))
        v_op = tf.nn.relu(solver)
        return v_op
    
    def matrix_loss(self, metric):
        # TODO: rewrite metric
        return metric(self.target_y, self.activation(self.logits))