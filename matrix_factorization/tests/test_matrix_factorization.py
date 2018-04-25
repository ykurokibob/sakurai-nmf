import numpy as np
import tensorflow as tf

from matrix_factorization.matrix_factorization import NonlinearSemiNMF, SemiNMF

np.random.seed(42)
tf.set_random_seed(42)

import time

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
    import scipy.io as sio
    mat = sio.loadmat('test_random.mat')
    a = mat['mat']
    with tf.Session() as sess:
        init.run()
        
        global_start_time = time.time()
        local_duration = 0
        
        num_iters = 10
        for _ in range(num_iters):
            
            time.sleep(0.5)
            
            start_time = time.time()
            _ = sess.run([u_op], feed_dict={mf.target_y: a})
            u_duration = time.time() - start_time
            local_duration += u_duration
            print('Duration compute u: {}'.format(u_duration))
            
            time.sleep(0.5)
            
            start_time = time.time()
            _ = sess.run([v_op], feed_dict={mf.target_y: a})
            v_duration = time.time() - start_time
            local_duration += v_duration
            print('Duration compute v: {}'.format(v_duration))
            
            time.sleep(0.5)
            
            start_time = time.time()
            loss = sess.run([loss_op], feed_dict={mf.target_y: a})
            loss_duration = time.time() - start_time
            local_duration += loss_duration
            print('Duration compute loss: {}, loss: {}'.format(loss_duration, loss))
        
        global_end_time = time.time()
        duration = global_end_time - global_start_time - (0.5 * 3 * num_iters) - local_duration
        print('Duration compute other process: {}'.format(duration))



def wrapper_nmf2(nmf_class, use_bias=False, minval=0):
    print("")
    target = tf.placeholder(tf.float32, [1000, 500])
    mf = nmf_class(target, (1000, 784), (784, 500), use_bias=use_bias)
    u_op, v_op, bias_op, loss_op = mf.factorize()
    init = tf.global_variables_initializer()
    import scipy.io as sio
    if minval == 0:
        mat = sio.loadmat('test_non_nega_random.mat')
        a = mat['mat']
    else:
        mat = sio.loadmat('test_random.mat')
        a = mat['mat']
    with tf.Session() as sess:
        init.run()
        
        global_start_time = time.time()
        local_duration = 0
        
        num_iters = 2
        for _ in range(num_iters):
            
            for _ in range(100):
                _ = sess.run([v_op], feed_dict={mf.target_y: a})
            _ = sess.run([u_op], feed_dict={mf.target_y: a})
            loss = sess.run(loss_op, feed_dict={mf.target_y: a})
            
            print('loss {}'.format(loss))
        
        global_end_time = time.time()
        duration = global_end_time - global_start_time
        print('Duration compute processes: {}'.format(duration))

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
    wrapper_nmf2(NonlinearSemiNMF, use_bias=False)


def test_nonlinear_semi_nmf_have_bias():
    wrapper_nmf2(NonlinearSemiNMF, use_bias=True)


def test_semi_nmf_no_bias():
    wrapper_nmf2(SemiNMF, use_bias=False)


def test_semi_nmf_have_bias():
    wrapper_nmf2(SemiNMF, use_bias=True)


def main(_):
    # Compute too much time.
    test_nonlinear_semi_nmf_no_bias()
    # test_nonlinear_semi_nmf_have_bias()
    
    test_semi_nmf_no_bias()
    
    # The loss would not be less.
    test_semi_nmf_have_bias()
 

if __name__ == '__main__':
    tf.app.run()