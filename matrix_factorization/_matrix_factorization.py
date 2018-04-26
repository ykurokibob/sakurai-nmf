import keras.backend.tensorflow_backend as K
import tensorflow as tf

from exception import MatrixFactorizationError
from losses import frobenius_norm
from matrix_factorization._algorithm import \
  _SemiNMF, _NonlinearSemiNMF, _BiasNonlinearSNMF, _BiasSemiNMF, MatrixFactorize


# TODO Replace v_shape and u_shape
class MatrixFactorizeInterface(object):

  def __init__(self,
               target_y,
               v_shape, u_shape,
               metric=frobenius_norm,
               rcond=1e-3,
               activation=tf.identity,
               use_bias=True,
               is_first=False,
               inputs=None,
               name=0):
    """
    :param target_y: Target Matrix factorized.
    :param v_shape: Weights Matrix's shape.
    :param u_shape: Non-negative Matrix's shape.
    :param metric: Evaluate relative residual.
    :param rcond: cutoff parameter of pinv
    :param activation: Activation Function
    :param use_bias: bool, whether use bias or not.
    :param is_first: is first layer or not.
    :param inputs: Use only at first layer.
    :param name: Pointer
    """
    self.target_y = target_y
    self.v_shape = v_shape
    self.u_shape = u_shape

    self.metric = metric
    self.rcond = rcond
    self.activation = activation
    self.use_bias = use_bias
    self.is_first = is_first
    self.name = self.__class__.__name__ + "_{0}".format(name)
    self.inputs = inputs

    # local matrix factorization steps.
    self.i = K.variable(0)

    # Matrix Factorization algorithm.
    self.algorithm = None

  def factorize(self):
    if not isinstance(self.algorithm, MatrixFactorize):
      raise MatrixFactorizationError("algorithm uninitialized.")

    # Define factorize operations.
    with tf.name_scope(self.name):
      with tf.name_scope('compute_u'):
        u_op, bias_op = self.algorithm.compute_u()

      with tf.name_scope('compute_v'):
        # If first layer, no need to compute v
        if not self.is_first:
          v_op = self.algorithm.compute_v(u_op)
        else:
          v_op = tf.no_op()

      # Calculate a loss.
      with tf.name_scope('factorize_metric'):
        loss = self.algorithm.matrix_loss(self.metric)

      tf.summary.histogram('u_op', u_op)
      tf.summary.histogram('v_op', v_op)
      if self.use_bias:
        tf.summary.histogram('bias_op', bias_op)
      tf.summary.scalar('fact_loss', loss)
      return u_op, v_op, bias_op, loss


class SemiNMF(MatrixFactorizeInterface):

  def __init__(self,
               target_y,
               v_shape, u_shape,
               **kwargs):
    super(SemiNMF, self).__init__(target_y, v_shape, u_shape, **kwargs)
    self.u_iter = 1
    self.v_iter = 1

    if not self.use_bias:
      self.algorithm = _SemiNMF(target_y=target_y,
                                u_shape=u_shape,
                                v_shape=v_shape,
                                name=self.name,
                                inputs=self.inputs)
    else:
      self.algorithm = _BiasSemiNMF(target_y=target_y,
                                    u_shape=u_shape,
                                    v_shape=v_shape,
                                    name=self.name,
                                    inputs=self.inputs)


class NonlinearSemiNMF(MatrixFactorizeInterface):

  def __init__(self,
               target_y,
               v_shape, u_shape,
               activation=tf.nn.relu,
               **kwargs):
    """
    :param activation: ReLU
    """
    super(NonlinearSemiNMF, self).__init__(target_y, v_shape, u_shape, activation=activation, **kwargs)
    self.u_iter = 10
    self.v_iter = 1

    if not self.use_bias:
      self.algorithm = _NonlinearSemiNMF(target_y=target_y,
                                         u_shape=u_shape,
                                         v_shape=v_shape,
                                         name=self.name,
                                         inputs=self.inputs)
    else:
      self.algorithm = _BiasNonlinearSNMF(target_y=target_y,
                                          u_shape=u_shape,
                                          v_shape=v_shape,
                                          name=self.name,
                                          inputs=self.inputs)
