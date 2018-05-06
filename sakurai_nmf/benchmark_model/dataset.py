import numpy as np


def load_data():
  x = np.linspace(-2, 2, 10000)
  y = np.sin(x)
  return x[..., np.newaxis], y[..., np.newaxis]
