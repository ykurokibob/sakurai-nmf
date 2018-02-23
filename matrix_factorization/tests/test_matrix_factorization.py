import numpy as np
import tensorflow as tf

from matrix_factorization.matrix_factorization import NonlinearSemiNMF, SemiNMF

np.random.seed(42)
tf.set_random_seed(42)


def test_initializer():
  print()
  target_y = tf.random_uniform(shape=(6000, 500), maxval=1)
  v_shape = (6000, 784)
  u_shape = (784, 500)
  print("not use bias")
  mf = NonlinearSemiNMF(target_y=target_y, v_shape=v_shape, u_shape=u_shape, use_bias=False, name='1')
  mf = SemiNMF(target_y=target_y, v_shape=v_shape, u_shape=u_shape, use_bias=False, name='2')
  print("use bias")
  mf = NonlinearSemiNMF(target_y=target_y, v_shape=v_shape, u_shape=u_shape, name='3')
  mf = SemiNMF(target_y=target_y, v_shape=v_shape, u_shape=u_shape, name='4')


def wrapper_nmf(nmf_class, use_bias=False, minval=0):
  print("")
  target = tf.placeholder(tf.float32, [1000, 500])
  mf = nmf_class(target, (1000, 784), (784, 500), use_bias=use_bias)
  u_op, v_op, bias_op, loss_op = mf.factorize()
  init = tf.global_variables_initializer()
  a = np.random.uniform(minval, 1, size=[1000, 500])
  with tf.Session() as sess:
    init.run()
    for i in range(10):
      loss, v = sess.run([loss_op, v_op], feed_dict={mf.target_y: a})
      print("loss", loss, "min(v)", np.min(v), "max(v)", np.max(v))


def nonlinear_semi_nmf():
  tf.reset_default_graph()
  target = tf.placeholder(tf.float32, [1000, 500])
  mf = NonlinearSemiNMF(target, (1000, 784), (784, 500), use_bias=True)
  u_op, v_op, bias_op, loss_op = mf.factorize()

  a = np.random.uniform(0, 1, size=[1000, 500])

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    init.run()
    for i in range(10):
      _, loss = sess.run([u_op, loss_op], feed_dict={mf.target_y: a})
      print("loss", loss)
    _, loss = sess.run([v_op, loss_op], feed_dict={mf.target_y: a})
    print("loss", loss)


def test_nonlinear_semi_nmf_no_bias():
  wrapper_nmf(NonlinearSemiNMF, use_bias=False)


def test_nonlinear_semi_nmf_have_bias():
  wrapper_nmf(NonlinearSemiNMF, use_bias=True)


def test_semi_nmf_no_bias():
  wrapper_nmf(SemiNMF, use_bias=False)


def test_semi_nmf_have_bias():
  wrapper_nmf(SemiNMF, use_bias=True)


def main(_):
  # test_semi_nmf_no_bias()
  # test_nonlinear_semi_nmf_no_bias()
  nonlinear_semi_nmf()


if __name__ == '__main__':
  tf.app.run()
