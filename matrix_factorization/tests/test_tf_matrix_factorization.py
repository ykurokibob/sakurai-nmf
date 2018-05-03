from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.io as sio
import tensorflow as tf

from losses import frobenius_norm, np_frobenius_norm
from matrix_factorization import nonlin_semi_nmf, semi_nmf
from matrix_factorization.utility import relu

mat_file = './np_tests/small_v_neg.mat'


def print_format(lib, algo, a, u, v, old_loss, new_loss, duration):
    print('\n[{}]Solve {}\n\t'
          'a {} u {} v {}\n\t'
          'old loss {}\n\t'
          'new loss {}\n\t'
          'process duration {}'.format(
        lib, algo,
        a.shape, u.shape, v.shape, old_loss, new_loss, duration))


def test_np_vanilla_semi_nmf():
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
    print_format('Numpy', 'semi-NMF', a, u, v, old_loss, new_loss, duration)


def test_np_biased_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    start_time = time.time()
    
    u, bias_v = semi_nmf(a, u, bias_v, use_bias=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
    
    new_loss = np_frobenius_norm(a, bias_u @ bias_v)
    assert a.shape == (bias_u @ bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('Numpy', 'biased semi-NMF', a, bias_u, bias_v, old_loss, new_loss, duration)


def test_np_vanilla_nonlin_semi_nmf():
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
    print_format('Numpy', 'Nonlinear semi-NMF', a, u, v, old_loss, new_loss, duration)


def test_np_not_calc_v_vanilla_nonlin_semi_nmf():
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
    print_format('Numpy', 'Nonlinear semi-NMF(NOT CALCULATE v)', a, u, v, old_loss, new_loss, duration)


def test_np_biased_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    start_time = time.time()
    
    u, bias_v = nonlin_semi_nmf(a, u, bias_v, use_bias=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
    
    new_loss = np_frobenius_norm(a, relu(bias_u @ bias_v))
    assert a.shape == (bias_u @ bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('Numpy', 'biased Nonlinear semi-NMF', a, bias_u, bias_v, old_loss, new_loss, duration)


def test_np_not_calc_v_biased_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    
    start_time = time.time()
    
    u, bias_v = nonlin_semi_nmf(a, u, bias_v, use_bias=True, num_calc_v=0)
    
    end_time = time.time()
    duration = end_time - start_time
    
    bias_u = np.hstack((u, np.ones((u.shape[0], 1))))
    
    new_loss = np_frobenius_norm(a, relu(bias_u @ bias_v))
    assert a.shape == (bias_u @ bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('Numpy', 'biased Nonlinear semi-NMF(NOT CALC v)', a, bias_u, bias_v, old_loss, new_loss, duration)


def test_tf_vanilla_semi_nmf():
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
    print_format('TensorFlow', 'semi-NMF', a, u, v, old_loss, new_loss, duration)


def test_tf_biased_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    bias_v_ph = tf.placeholder(tf.float64, shape=bias_v.shape)
    
    tf_bias_u, tf_v = semi_nmf(a_ph, u_ph, bias_v_ph, use_bias=True, use_tf=True)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _bias_v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, u_ph: u, bias_v_ph: bias_v})
        end_time = time.time()
    
    duration = end_time - start_time
    _bias_u = np.hstack((_u, np.ones((_u.shape[0], 1))))
    new_loss = np_frobenius_norm(a, _bias_u @ _bias_v)
    assert a.shape == (_bias_u @ _bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('TensorFlow', 'biased semi-NMF', a, _bias_u, _bias_v, old_loss, new_loss, duration)


def test_tf_nonlin_semi_nmf():
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
    print_format('TensorFlow', 'Nonlinear semi-NMF', a, u, v, old_loss, new_loss, duration)


def test_tf_not_calc_v_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    # [1000, 500]
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    # [1000, 201]
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    # [200, 500]
    v_ph = tf.placeholder(tf.float64, shape=v.shape)
    tf_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, v_ph, use_tf=True, use_bias=False, num_calc_v=0, num_calc_u=1)
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
    print_format('TensorFlow', 'Nonlinear semi-NMF(NOT CALCLATE v)', a, u, v, old_loss, new_loss, duration)


def test_tf_biased_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    bias_v_ph = tf.placeholder(tf.float64, shape=bias_v.shape)
    
    tf_bias_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, bias_v_ph, use_bias=True, use_tf=True)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _bias_v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, u_ph: u, bias_v_ph: bias_v})
        end_time = time.time()
    
    duration = end_time - start_time
    _bias_u = np.hstack((_u, np.ones((_u.shape[0], 1))))
    new_loss = np_frobenius_norm(a, relu(_bias_u @ _bias_v))
    assert a.shape == (_bias_u @ _bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('TensorFlow', 'biased Nonlinear semi-NMF', a, _bias_u, _bias_v, old_loss, new_loss, duration)


def test_tf_not_calc_v_biased_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    a, u, v = auv['a'], auv['u'], auv['v']
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    bias_v_ph = tf.placeholder(tf.float64, shape=bias_v.shape)
    
    tf_bias_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, bias_v_ph, use_bias=True, use_tf=True, num_calc_v=0)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _bias_v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, u_ph: u, bias_v_ph: bias_v})
        end_time = time.time()
    
    duration = end_time - start_time
    _bias_u = np.hstack((_u, np.ones((_u.shape[0], 1))))
    new_loss = np_frobenius_norm(a, relu(_bias_u @ _bias_v))
    assert a.shape == (_bias_u @ _bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('TensorFlow', 'biased Nonlinear semi-NMF(NOT CALC v)', a, _bias_u, _bias_v, old_loss, new_loss,
                 duration)


def test_original_biased_nonlin_semi_nmf():
    auv = sio.loadmat(mat_file)
    u, v = auv['u'], auv['v']
    a = relu(u @ v)
    bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    bias_v_ph = tf.placeholder(tf.float64, shape=bias_v.shape)
    
    tf_bias_u, tf_v = nonlin_semi_nmf(a_ph, u_ph, bias_v_ph, use_bias=True, use_tf=True, num_calc_v=0)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _bias_v = sess.run([tf_bias_u, tf_v], feed_dict={a_ph: a, u_ph: u, bias_v_ph: bias_v})
        end_time = time.time()
    
    duration = end_time - start_time
    _bias_u = np.hstack((_u, np.ones((_u.shape[0], 1))))
    new_loss = np_frobenius_norm(a, relu(_bias_u @ _bias_v))
    assert a.shape == (_bias_u @ _bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print_format('TensorFlow', 'biased Nonlinear semi-NMF(NOT CALC v)', a, _bias_u, _bias_v, old_loss, new_loss,
                 duration)