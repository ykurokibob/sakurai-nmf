import numpy as np
import tensorflow as tf


def frobenius_norm(x, y):
  losses = tf.subtract(x, y)
  losses = tf.divide(losses, tf.norm(y))
  # TODO: HOW TO USE FROBENIOUS
  loss = tf.square(tf.norm(losses))
  return loss


def np_frobenius_norm(x, y):
  return np.linalg.norm(x - y) / np.linalg.norm(y)
