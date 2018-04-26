# Nonlinear semi-NMF for optimizer.

## Implement

+ [x] Vanilla NMF with TF and Numpy.
+ [x] **NOT convergence to epslion(<1e-8).**
+ [x] Use this Nonlinear-sNMF for NMF-NeuralNets.
+ [x] faster...(so far, fucking slow)

### biased Nonlinear semi-NMF with TensorFlow

```python
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
 ```


```python
solve Nonlinear semi-NMF
	old loss 15270.320488460004
	new loss 0.7154995132441608
	process duration 0.544182300567627
```

