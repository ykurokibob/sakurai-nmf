import tensorflow as tf

from model import DenseModel


def main(_):
  model = DenseModel()
  print("inputs.dtype", model.inputs)
  print("output.dtype", model.outputs)
  print("label.dtype", model.labels)


if __name__ == '__main__':
  tf.app.run()
