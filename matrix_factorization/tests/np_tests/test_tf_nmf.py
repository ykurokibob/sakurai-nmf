import time

import scipy.io as sio
import tensorflow as tf

from losses import frobenius_norm, np_frobenius_norm
from matrix_factorization.np_nmf import nonlin_semi_nmf, semi_nmf


def test_semi_nmf():
    auv = sio.loadmat('auv.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    v_ph = tf.placeholder(tf.float64, shape=v.shape)
    tf_u, tf_v = tf.py_func(semi_nmf, [a_ph, u_ph, v_ph], [tf.float64, tf.float64])
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _v = sess.run([tf_u, tf_v], feed_dict={a_ph: a, u_ph: u, v_ph: v})
        end_time = time.time()
    
    duration = end_time - start_time
    new_loss = np_frobenius_norm(a, _u @ _v)
    assert a.shape == (_u @ _v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_large_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    v_ph = tf.placeholder(tf.float64, shape=v.shape)
    tf_u, tf_v = tf.py_func(semi_nmf, [a_ph, u_ph, v_ph], [tf.float64, tf.float64])
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _v = sess.run([tf_u, tf_v], feed_dict={a_ph: a, u_ph: u, v_ph: v})
        end_time = time.time()
    
    duration = end_time - start_time
    new_loss = np_frobenius_norm(a, _u @ _v)
    assert a.shape == (_u @ _v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_u_neg_nonlin_semi_nmf():
    auv = sio.loadmat('u_neg.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    old_loss = np_frobenius_norm(a, u @ v)
    
    a_ph = tf.placeholder(tf.float64, shape=a.shape)
    u_ph = tf.placeholder(tf.float64, shape=u.shape)
    v_ph = tf.placeholder(tf.float64, shape=v.shape)
    tf_u, tf_v = tf.py_func(nonlin_semi_nmf, [a_ph, u_ph, v_ph], [tf.float64, tf.float64])
    tf_loss = frobenius_norm(a_ph, tf.nn.relu(tf_u @ tf_v))
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        
        start_time = time.time()
        _u, _v, new_loss = sess.run([tf_u, tf_v, tf_loss], feed_dict={a_ph: a, u_ph: u, v_ph: v})
        end_time = time.time()
    
    duration = end_time - start_time
    assert a.shape == (_u @ _v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve Nonlinear semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))