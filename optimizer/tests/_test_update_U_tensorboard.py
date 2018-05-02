import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.utils.np_utils import to_categorical

from losses import frobenius_norm
from model import DenseModel, BackpropModel
from optimizer import NMF

tf.set_random_seed(42)


def test_update_u_tensorboard():
  print()
  with tf.device("/device:GPU:0"):
    model = DenseModel(batch_size=3000, use_gpu=True)
    bar = tf.get_default_graph().get_operation_by_name('DenseModel/layers/dense/Relu')
    opt = NMF(mf_iteration=2)
    model.compile(opt, frobenius_norm, 'categorical')
  (x_train, y_train), (x_test, y_test) = load_data()
  x_train = x_train.astype('float32').reshape((-1, 784)) / 255.
  x_test = x_test.astype('float32').reshape((-1, 784)) / 255.
  y_train = to_categorical(y_train)
  model.fit(x_train, y_train, mf_epochs=1, bp_epochs=3, use_nmf=True, use_backprop=True)


def test_backprop():
  print()
  model = BackpropModel()
  (x_train, y_train), (x_test, y_test) = load_data()
  x_train = x_train.astype('float32').reshape((-1, 784)) / 255.
  x_test = x_test.astype('float32').reshape((-1, 784)) / 255.
  y_train = to_categorical(y_train)
  model.train(x_train, y_train)


if __name__ == '__main__':
  test_update_u_tensorboard()
