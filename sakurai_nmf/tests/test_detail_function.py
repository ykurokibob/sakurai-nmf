from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import matrix_factorization as mf


class TestDetailFunction(tf.test.TestCase):
    def test_check_shape(self):
        print()
        a = tf.random_uniform((100, 50))
        u = tf.random_uniform((100, 25))
        v = tf.random_uniform((25, 50))
        tf_u, tf_v = mf.semi_nmf(a, u, v, use_tf=True)
        print(tf_u, tf_v)
        assert u.shape == tf_u.shape
        assert v.shape == tf_v.shape
        