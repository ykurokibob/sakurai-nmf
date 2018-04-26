# Nonlinear semi-NMF for optimizer.

## Implement

+ [x] Vanilla NMF with TF and Numpy.
+ [x] **NOT convergence to epslion(<1e-8).**
+ [x] Use this Nonlinear-sNMF for NMF-NeuralNets.
+ [x] faster...(so far, fucking slow)

### Example Nonlinear semi-NMF

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

### Results of test_matrix_factorization.py

```python
[Numpy]Solve semi-NMF
	old loss 338.7211354286444
	new loss 2.531794131273001e-15
	process duration 0.22645926475524902
.
[Numpy]Solve Nonlinear semi-NMF
	old loss 338.7211354286444
	new loss 0.0002981372559180241
	process duration 0.18893766403198242
.
[Numpy]Solve biased semi-NMF
	old loss 338.7211354286444
	new loss 8.872548363465481e-14
	process duration 0.37479066848754883
.
[Numpy]Solve biased Nonlinear semi-NMF
	old loss 338.7211354286444
	new loss 0.013125810643673044
	process duration 0.518439531326294
.
[TensorFlow]Solve semi-NMF
	old loss 338.7211354286444
	new loss 2.5985280482743766e-15
	process duration 0.34461069107055664
.
[TensorFlow]Solve biased semi-NMF
	old loss 338.7211354286444
	new loss 0.0002981372559180241
	process duration 0.20970702171325684
.
[TensorFlow]Solve Nonlinear semi-NMF
	old loss 338.7211354286444
	new loss 8.874093318886518e-14
	process duration 0.3893768787384033
.
[TensorFlow]Solve biased Nonlinear semi-NMF
	old loss 338.7211354286444
	new loss 0.013125810643673044
	process duration 0.5391981601715088
```

