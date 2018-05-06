import logging

import numpy as np

epsilon = 1e-6

log_levels = dict(
    info=logging.INFO,
    debug=logging.DEBUG,
    critical=logging.CRITICAL,
    warning=logging.WARNING,
    fatal=logging.FATAL
)


def batch(x, y, batch_size):
  rand_index = np.random.choice(len(x), batch_size)
  return x[rand_index], y[rand_index]


def prod(xs):
  """Computes the product along the elements in an iterable. Returns 1 for empty iterable.

  Args:
      xs: Iterable containing numbers.

  Returns: Product along iterable.

  """
  p = 1
  for x in xs:
    p *= x
  return p


def rank(x):
  return x.get_shape().ndims


def shape(x, unknown=-1):
  return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())
