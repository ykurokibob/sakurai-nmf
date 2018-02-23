import numpy as np
import tensorflow as tf


def prod(xs):
  """Computes the product along the elements in an iterable. Returns 1 for empty iterable.
  Args:
      xs: Iterable containing numbers.
  Returns: Product along iterable.
  """
  p = 1
  for x in xs:
    p *= x
  return p


def shape(x, unknown=-1):
  return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


def _makearray(a):
  new = np.asarray(a)
  wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
  return new, wrap


def _isEmpty2d(arr):
  # check size first for efficiency
  return shape(arr) == 0 and prod(arr.shape[-2:]) == 0


def _pinv(a, rcond=1e-8):
  a, wrap = _makearray(a)
  rcond = np.asarray(rcond)
  # if _isEmpty2d(a):
  #   res = np.empty(a.shape[:-2] + (a.shape[-1], a.shape[-2]), dtype=a.dtype)
  #   return wrap(res)
  # a = a.conjugate()
  u, s, vt = np.linalg.svd(a, full_matrices=False)

  # discard small singular values
  cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
  large = s > cutoff
  s = np.divide(1, s, where=large, out=s)
  s[~large] = 0

  res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
  print('res', res)
  return wrap(res)


def pinv(inputs):
  return tf.py_func(_pinv, [inputs], tf.float32)


def tf_pinv(a, rcond=1e-8):
  with tf.name_scope("tf_pinv"):
    if _isEmpty2d(a):
      return a
    s, u, v = tf.svd(a, full_matrices=False)
    vt = tf.transpose(v)

    # discard small singular values
    cutoff = rcond * tf.reduce_max(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = tf.divide(1, s)

    zeros = tf.zeros_like(s)
    s = tf.where(large, s, zeros)

    res = tf.matmul(tf.transpose(vt), tf.multiply(s[..., tf.newaxis], tf.transpose(u)))
    return res