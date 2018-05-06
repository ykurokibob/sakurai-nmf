# Nonlinear semi-NMF for optimizer.

## Implement

+ [x] Vanilla NMF with TF and Numpy.
+ [x] **NOT convergence to epslion(<1e-8).**
+ [x] Use this Nonlinear-sNMF for NMF-NeuralNets.
+ [x] faster...(so far, fucking slow)

### Example Nonlinear semi-NMF

```python
def test_tf_biased_nonlin_semi_nmf():
    auv = sio.loadmat('./np_tests/large.mat')
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
    new_loss = np_frobenius_norm(a, _bias_u @ _bias_v)
    assert a.shape == (_bias_u @ _bias_v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('\n[TensorFlow]Solve biased Nonlinear semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))
 ```

### Results of test_matrix_factorization.py

```python
$TF_CPP_MIN_LOG_LEVEL=3 pytest -s test_matrix_factorization.py
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

