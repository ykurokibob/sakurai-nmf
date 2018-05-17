# NMF Neural Nets

The NMF optimizer doesn't work well. But, the NMF would get well.


## Results of MNIST

Now it cannot run well. Please see [test_optimizers.py](https://github.com/ashigirl96/nmf-nn/blob/master/optimizer/tests/test_optimizers.py) or [mnist.py](https://github.com/ashigirl96/nmf-nn/blob/master/examples/mnist.py).

But now we can get accuracy 84% for only 6 iterations without ReLU.

```python
$ python sakurai_nmf/examples/mnist.py --num_bp_iters 5 --num_mf_iters 1
Using TensorFlow backend.
NMF-optimizer
(1/1) [Train]loss 1.880, accuracy 85.533 [Test]loss 1.902, accuracy 81.473
Adam-optimizer
(5/5) [Train]loss 0.629, accuracy 84.167 [Test]loss 0.624, accuracy 84.267
```

Use only Nonlinear semi-NMF.

```python
$ python sakurai_nmf/examples/mnist.py --num_bp_iters 0 --num_mf_iters 10 --use_relu
Using TensorFlow backend.
NMF-optimizer
(10/10) [Train]loss 1.673, accuracy 87.400 [Test]loss 1.709, accuracy 81.527
```


## Results of Fashion MNIST


we can get accuracy 74% for only 17 iterations.


```pythonstub
NMF-optimizer
(2/2) [Train]loss 1.9780992698087023, accuracy 50.19999694824219 [Test]loss 2.0002613045103814, accuracy 64.82400360107422
Adam-optimizer
(15/15) [Train]loss 0.8079785835028875, accuracy 71.73999786376953 [Test]loss 0.750417619892654, accuracy 74.11599884033203

```
