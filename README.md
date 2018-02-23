# NMF Neural Nets

The NMF optimizer doesn't work well. But, the NMF would get well.


```python
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
```

```
loss 127.94389
loss 0.7218605
loss 0.5472945
loss 0.489057
loss 0.45049044
loss 0.41869712
loss 0.39025217
loss 0.36473152
loss 0.3411206
loss 0.3186099
loss 0.29780647
```