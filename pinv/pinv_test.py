# import tfe
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from losses import frobenius_norm
from pinv import tf_pinv

# tfe.enable_eager_execution()

A2 = np.array([[-1.4063588, -0.4529773, 1.0436655, -1.3272166, -1.2746028],
               [-0.4117019, 1.1795201, 0.1849949, 1.1964049, 0.64962494],
               [-0.3475104, 2.1557577, 1.5274503, -0.5880765, 0.8736929],
               [-0.89176613, 1.1701676, -0.73233956, -2.2032936, 1.27408],
               [-1.5731467, -2.0532212, 0.99725246, -0.6638079, 1.0961444],
               [0.4248257, -0.97618026, 1.450069, -1.0241394, 0.7442274],
               [-0.81716555, 2.6097229, -0.12604012, -0.6079891, 0.41683516]])

"""
A1 = np.array([[7.7778284e-11, 7.0281385e-11, 4.6332944e-11, 5.3424647e-11],
               [4.4646811e-11, 7.1549537e-11, 5.0201964e-11, 4.8699274e-11],
               [1.0943807e-11, 2.6453471e-11, 5.1286673e-11, 8.7640166e-11],
               [3.4845293e-11, 2.2363352e-11, 1.9306778e-11, 7.4436013e-11],
               [3.0471421e-11, 2.2022320e-11, 9.7882631e-12, 4.7243154e-11]],
              dtype=np.float32)
"""



def tfe_main():
  # tf_pinv(tfe.Variable(A1))
  tf_pinv(tfe.Variable(A2))
  # with tf.Session() as sess:
  #   init.run()
  #   print(rcond.name)
  #   print(sess.run(a_inv))
  #   print(sess.run(b))
  #   print(sess.run(rcond))




def tf_main():
  a1 = tf.random_uniform([784, 784], -100, 100)
  # a1 = tf.Variable(A1)
  _a1_inv = tf_pinv(a1)
  # na1_inv = np.linalg.pinv(A1)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    init.run()
    a1_inv, ta = sess.run([_a1_inv, a1])
    np.testing.assert_almost_equal(a1_inv @ ta, np.eye(ta.shape[0]))
    # np.testing.assert_almost_equal(np.linalg.pinv(ta) @ ta, np.eye(ta.shape[0]))


def test_tf_pinv():
    a1 = tf.random_uniform([50, 100], -1, 1)
    import scipy.io as sio
    a1 = tf.constant(sio.loadmat('test_random.mat')['mat'])
    
    a1_inv = tf_pinv(a1)
    
    muled_a = a1 @ a1_inv @ a1
    loss = frobenius_norm(a1, muled_a)
    print('loss', loss)


def main(_):
    test_tf_pinv()

if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run()