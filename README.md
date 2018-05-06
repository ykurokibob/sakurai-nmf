# Sakurai NMF 

This library is used as `(Nonlinear) Semi Non-negative Matrix Factorization` and neural network optimization using `(Nonlinear) Semi-NMF`.

The NMF optimizer doesn't work well. But, the NMF would get well.

## Install

```
pip install -e git+https://github.com/ashigirl96/sakurai-nmf.git#egg=sakurai-nmf
```

## Example 

```python
from sakurai_nmf.matrix_factorization import semi_nmf
from sakurai_nmf.losses import np_frobenius_norm
import numpy as np
a = np.random.uniform(size=(100, 300)).astype(np.float64)
u = np.random.uniform(-1., 1., size=[100, 10]).astype('float64')
v = np.random.uniform(0., 1., size=[10, 300]).astype('float64')
_u, _v = semi_nmf(a, u, v)

>>> np_frobenius_norm(a, u @ v)
2.1015510101001085

>>> np_frobenius_norm(a, _u @ _v)
0.8731707199786362
```


## Results of MNIST

Now it cannot run well. Please see [(fashion)mnist.py](https://github.com/ashigirl96/sakurai-nmf/blob/master/sakurai_nmf/examples/mnist.py).

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


```python
NMF-optimizer
(2/2) [Train]loss 1.9780992698087023, accuracy 50.19999694824219 [Test]loss 2.0002613045103814, accuracy 64.82400360107422
Adam-optimizer
(15/15) [Train]loss 0.8079785835028875, accuracy 71.73999786376953 [Test]loss 0.750417619892654, accuracy 74.11599884033203
```
