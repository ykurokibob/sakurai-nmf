from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def frobenius_norm(original_x, muled_y):
    losses = original_x - muled_y
    losses = tf.norm(losses) / tf.norm(original_x)
    return losses


def np_frobenius_norm(x, y):
  return np.linalg.norm(x - y) / np.linalg.norm(y)
