# NMF Neural Nets

The NMF optimizer doesn't work well. But, the NMF would get well.


## Results of MNIST

Now it cannot run well. Please see [test_optimizers.py](https://github.com/ashigirl96/nmf-nn/blob/master/optimizer/tests/test_optimizers.py) or [mnist.py](https://github.com/ashigirl96/nmf-nn/blob/master/examples/mnist.py).

But now we can get accuracy 83% for only 7 iterations.

```python
NMF-optimizer
(2/2) [Test]loss 2.099674575219118, accuracy 49.67333297729492
==========
Adam-optimizer
(5/5) [Test]loss 0.5333448813711203, accuracy 83.63999938964844
```

## Results of Fashion MNIST


we can get accuracy 74% for only 17 iterations.


```pythonstub
NMF-optimizer
(2/2) [Train]loss 1.9780992698087023, accuracy 50.19999694824219 [Test]loss 2.0002613045103814, accuracy 64.82400360107422
Adam-optimizer
(15/15) [Train]loss 0.8079785835028875, accuracy 71.73999786376953 [Test]loss 0.750417619892654, accuracy 74.11599884033203

```
