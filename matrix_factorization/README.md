# Nonlinear semi-NMF for optimizer.

## Implement

+ [x] Vanilla NMF with TF and Numpy.
+ [ ] **NOT convergence to epslion(<1e-8).**
+ [ ] Use this Nonlinear-sNMF for NMF-NeuralNets.
+ [ ] faster...(so far, fucking slow)

### biased Nonlinear semi-NMF with TensorFlow

```python
target = K.variable(tf.random_uniform(shape=(6000, 10), maxval=1), name="y_target")
mf = BiasNSNMF(target, (6000, 100), (100, 10))
mf2 = BiasNSNMF(mf.u, (6000, 784), (784, 100))
```


```python
$ python test_mf_bias_nsnmf.py         *[master]
BiasNSNMF.loss = 0.9182373285293579 / 0
BiasNSNMF.loss = 0.7949256300926208 / 1
BiasNSNMF.loss = 0.5727155804634094 / 2
BiasNSNMF.loss = 0.45173755288124084 / 3
BiasNSNMF.loss = 0.41608870029449463 / 4
BiasNSNMF.loss = 0.4731375277042389 / 5
BiasNSNMF.loss = 0.477230429649353 / 6
BiasNSNMF.loss = 0.5276081562042236 / 7
BiasNSNMF.loss = 0.5181043148040771 / 8
BiasNSNMF.loss = 0.554600715637207 / 9
End...
BiasNSNMF.loss = 1.0434532165527344 / 0
BiasNSNMF.loss = 1.1263998746871948 / 1
BiasNSNMF.loss = 1.4292962551116943 / 2
BiasNSNMF.loss = 1.8919872045516968 / 3
BiasNSNMF.loss = 2.3239381313323975 / 4
BiasNSNMF.loss = 2.7060446739196777 / 5
BiasNSNMF.loss = 3.077836751937866 / 6
BiasNSNMF.loss = 3.443369150161743 / 7
BiasNSNMF.loss = 3.8079569339752197 / 8
BiasNSNMF.loss = 4.1690897941589355 / 9
End...
```

