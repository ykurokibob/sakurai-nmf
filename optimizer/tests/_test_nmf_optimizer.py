import keras.backend.tensorflow_backend as K
import numpy as np
from keras.datasets.mnist import load_data
from keras.losses import mean_squared_error
from keras.utils.np_utils import to_categorical

from losses import frobenius_norm
from model import DenseModel
from optimizer import NMF

old_session = K.get_session()


def test_initializer():
  print("")
  dense_model = DenseModel()
  opt = NMF()


def test_backprop():
  print("")
  dense_model = DenseModel(batch_size=100)
  opt = NMF()
  dense_model.compile(opt, frobenius_norm, bp_metric=mean_squared_error)
  (x_train, y_train), (_, _) = load_data()
  x_train = x_train.astype(np.float32).reshape((-1, 784)) / 255
  y_train = to_categorical(y_train).astype(np.float32)
  print("y_train", y_train.dtype, y_train.shape)

  dense_model.fit(x=x_train, y=y_train, use_backprop=False, use_nmf=True, mf_epochs=2)


if __name__ == '__main__':
  test_backprop()
