# NMF Neural Nets

The NMF optimizer doesn't work well. But, the NMF would get well.


## Results of MNIST

Now it cannot run well. Please see [test_optimizers.py](https://github.com/ashigirl96/sakurai-nmf/blob/master/sakurai_nmf/tests/test_optimizers.py) or [mnist.py](https://github.com/ashigirl96/sakurai-nmf/blob/master/sakurai_nmf/examples/mnist.py).

But now we can get accuracy 84% for only 6 iterations without ReLU.

```bash
$ python sakurai_nmf/examples/mnist.py --num_bp_iters 5 --num_mf_iters 1
Using TensorFlow backend.
NMF-optimizer
(1/1) [Train]loss 1.880, accuracy 85.533 [Test]loss 1.902, accuracy 81.473
Adam-optimizer
(5/5) [Train]loss 0.629, accuracy 84.167 [Test]loss 0.624, accuracy 84.267
```

Use only Nonlinear semi-NMF.

```bash
$ python sakurai_nmf/examples/mnist.py --num_bp_iters 0 --num_mf_iters 10 --use_relu
Using TensorFlow backend.
NMF-optimizer
(10/10) [Train]loss 1.673, accuracy 87.400 [Test]loss 1.709, accuracy 81.527
```


## Results of Fashion MNIST


we can get accuracy 74% for only 3 iterations.


```bash
$ python sakurai_nmf/examples/mnist.py --num_bp_iters 0 --num_mf_iters 3 --dataset fashion --use_bias
Using TensorFlow backend.
NMF-optimizer
(3/3) [Train]loss 1.721, accuracy 90.400 [Test]loss 1.806, accuracy 74.873
```
