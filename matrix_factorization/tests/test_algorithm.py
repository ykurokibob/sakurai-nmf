import tensorflow as tf

from matrix_factorization.algorithm import MatrixFactorize
from matrix_factorization.algorithm import _BiasNonlinearSNMF


def test_matrix_fact():
  print()
  target_y = tf.random_uniform(shape=(6000, 500), maxval=1)
  v_shape = (6000, 784)
  u_shape = (784, 500)
  mf = MatrixFactorize(target_y=target_y, v_shape=v_shape, u_shape=u_shape, name="0")


def test_bias_matrix_fact():
  print()
  target_y = tf.random_uniform(shape=(6000, 500), maxval=1)
  v_shape = (6000, 784)
  u_shape = (784, 500)
  mf = _BiasNonlinearSNMF(target_y=target_y, v_shape=v_shape, u_shape=u_shape, name="1")
