from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from pathlib import Path

import functools
import numpy as np
import scipy.io as sio
import tensorflow as tf

from sakurai_nmf.losses import frobenius_norm, np_frobenius_norm
from sakurai_nmf.matrix_factorization import nonlin_semi_nmf, semi_nmf
from sakurai_nmf.matrix_factorization.utility import relu

python_path = Path(__file__).parent.joinpath('datasets')
mat_file = python_path.joinpath('./large.mat').as_posix()


def mat2tf_format(a, u, v):
    a = a.T
    _u = v.T
    v = u.T
    u = _u
    return a, u, v


semi_nmf = functools.partial(semi_nmf, data_format=False)
nonlin_semi_nmf = functools.partial(nonlin_semi_nmf, data_format=False)


class TestMatrixFactorization(tf.test.TestCase):
    def test_np_vanilla_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        start_time = time.time()
        
        u, v = semi_nmf(a, u, v, use_bias=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        new_loss = np_frobenius_norm(a, u @ v)
        assert a.shape == (u @ v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_np_biased_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        u, v = semi_nmf(a, u, v, use_bias=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        
        new_loss = np_frobenius_norm(a, u @ bias_v)
        assert a.shape == (u @ bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve biased semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_np_vanilla_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        start_time = time.time()
        
        u, v = nonlin_semi_nmf(a, u, v, use_bias=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        new_loss = np_frobenius_norm(a, relu(u @ v))
        assert a.shape == (u @ v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_np_not_calc_v_vanilla_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        start_time = time.time()
        
        u, v = nonlin_semi_nmf(a, u, v, use_bias=False, num_calc_v=0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        new_loss = np_frobenius_norm(a, relu(u @ v))
        assert a.shape == (u @ v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve Nonlinear semi-NMF(NOT CALCULATE v)\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_np_biased_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        biased_u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        biased_u, v = nonlin_semi_nmf(a, biased_u, v, use_bias=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        
        new_loss = np_frobenius_norm(a, relu(biased_u @ bias_v))
        assert a.shape == (biased_u @ bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve biased Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_np_not_calc_v_biased_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        biased_u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        biased_u, v = nonlin_semi_nmf(a, biased_u, v, use_bias=True, num_calc_v=0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        
        new_loss = np_frobenius_norm(a, relu(biased_u @ bias_v))
        assert a.shape == (biased_u @ bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[Numpy]Solve biased Nonlinear semi-NMF(NOT CALCULATE v)\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_vanilla_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        # [1000, 500]
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        # [1000, 201]
        u_ph = tf.placeholder(tf.float64, shape=u.shape)
        # [200, 500]
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        tf_u, tf_v = semi_nmf(a_ph, u_ph, v_ph, use_tf=True, use_bias=False)
        tf_loss = frobenius_norm(a_ph, tf.matmul(tf_u, tf_v))
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _u, _v, new_loss = sess.run([tf_u, tf_v, tf_loss], feed_dict={a_ph: a, u_ph: u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        assert a.shape == (_u @ _v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_biased_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
        old_loss = np_frobenius_norm(a, u @ v)
        
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        bias_u_ph = tf.placeholder(tf.float64, shape=bias_u.shape)
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        
        tf_bias_u, tf_v = semi_nmf(a_ph, bias_u_ph, v_ph, use_bias=True, use_tf=True)
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _bias_u, _v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, bias_u_ph: bias_u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        _bias_v = np.vstack((_v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, _bias_u @ _bias_v)
        assert a.shape == (_bias_u @ _bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve biased semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        # [1000, 500]
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        # [1000, 201]
        u_ph = tf.placeholder(tf.float64, shape=u.shape)
        # [200, 500]
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        tf_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, v_ph, use_tf=True, use_bias=False)
        tf_loss = frobenius_norm(a_ph, tf.nn.relu(tf.matmul(tf_u, tf_v)))
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _u, _v, new_loss = sess.run([tf_u, tf_v, tf_loss], feed_dict={a_ph: a, u_ph: u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        assert a.shape == (_u @ _v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_not_calc_v_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        # [1000, 500]
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        # [1000, 201]
        u_ph = tf.placeholder(tf.float64, shape=u.shape)
        # [200, 500]
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        tf_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, v_ph, use_tf=True, use_bias=False, num_calc_v=0)
        tf_loss = frobenius_norm(a_ph, tf.nn.relu(tf.matmul(tf_u, tf_v)))
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _u, _v, new_loss = sess.run([tf_u, tf_v, tf_loss], feed_dict={a_ph: a, u_ph: u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        assert a.shape == (_u @ _v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve Nonlinear semi-NMF(NOT CALCULATE v)\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_biased_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
        old_loss = np_frobenius_norm(a, u @ v)
        
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        bias_u_ph = tf.placeholder(tf.float64, shape=bias_u.shape)
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        
        tf_bias_u, tf_v = nonlin_semi_nmf(a_ph, bias_u_ph, v_ph,
                                          num_calc_v=1,
                                          use_bias=True, use_tf=True)
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _bias_u, _v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, bias_u_ph: bias_u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        _bias_v = np.vstack((_v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, relu(_bias_u @ _bias_v))
        assert a.shape == (_bias_u @ _bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve biased Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_tf_not_calc_v_biased_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
        old_loss = np_frobenius_norm(a, u @ v)
        
        a_ph = tf.placeholder(tf.float64, shape=a.shape)
        bias_u_ph = tf.placeholder(tf.float64, shape=bias_u.shape)
        v_ph = tf.placeholder(tf.float64, shape=v.shape)
        
        tf_bias_u, tf_v = nonlin_semi_nmf(a_ph, bias_u_ph, v_ph,
                                          num_calc_v=0,
                                          use_bias=True, use_tf=True)
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            start_time = time.time()
            _bias_u, _v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, bias_u_ph: bias_u, v_ph: v})
            end_time = time.time()
        
        duration = end_time - start_time
        _bias_v = np.vstack((_v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, relu(_bias_u @ _bias_v))
        assert a.shape == (_bias_u @ _bias_v).shape
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('\n[TensorFlow]Solve biased Nonlinear semi-NMF(NOT CALCULATE v)\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))