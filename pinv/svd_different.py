import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

A = np.array([[-1.4063588, -0.4529773, 1.0436655, -1.3272166, -1.2746028],
              [-0.4117019, 1.1795201, 0.1849949, 1.1964049, 0.64962494],
              [-0.3475104, 2.1557577, 1.5274503, -0.5880765, 0.8736929],
              [-0.89176613, 1.1701676, -0.73233956, -2.2032936, 1.27408],
              [-1.5731467, -2.0532212, 0.99725246, -0.6638079, 1.0961444],
              [0.4248257, -0.97618026, 1.450069, -1.0241394, 0.7442274],
              [-0.81716555, 2.6097229, -0.12604012, -0.6079891, 0.41683516]])


def np_main():
  u, s, vt = np.linalg.svd(A, full_matrices=False)
  return u, s, vt


def tf_main():
  a = tfe.Variable(A, trainable=False)
  s, u, v = tf.svd(a, full_matrices=False, compute_uv=True)
  vt = tf.transpose(v)
  return u, s, vt


if __name__ == '__main__':
  nu, ns, nvt = np_main()
  na = nu @ np.diag(ns) @ nvt

  tu, ts, tvt = tf_main()
  ta = tu @ tf.diag(ts) @ tvt

  np.testing.assert_almost_equal(na, ta)
