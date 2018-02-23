import keras.backend.tensorflow_backend as K
import numpy as np
import tensorflow as tf
from keras.metrics import categorical_accuracy, binary_accuracy, sparse_categorical_accuracy
from keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy

from optimizer.optimizers import NMF
from utils import batch


class Model(object):
  def __init__(self, batch_size=128, name=None, use_gpu=True):
    self.optimizer = None
    self.inputs = None
    self.labels = None
    self.name = name or self.__class__.__name__
    self.model = None
    self.loss = None
    self.train_op = None
    self.outputs = None
    self.accuracy = None
    self.batch_size = batch_size
    self.hiddens = [None]  # Not use first Layer.
    self.iter_size = []
    self.use_gpu = use_gpu
    self._build_model(batch_size)

  def _build_model(self, batch_size=128):
    raise NotImplementedError

  def compile(self, mf_opt: NMF, mf_metric, bp_metric=None):
    # TODO: Implement more incredible
    if bp_metric is 'categorical':
      bp_metric = categorical_crossentropy
      accuracy = categorical_accuracy
    elif bp_metric is 'binary':
      bp_metric = binary_crossentropy
      accuracy = binary_accuracy
    elif bp_metric is 'sparse_categorical':
      bp_metric = sparse_categorical_crossentropy
      accuracy = sparse_categorical_accuracy
    else:
      accuracy = categorical_accuracy

    self.optimizer = mf_opt
    assert self.labels is not None, "Please set the last layer to self.labels"

    self.optimizer.init_matrix(labels=self.labels, hiddens=self.hiddens)
    self.optimizer.loss_func = mf_metric

    # For Back propagation.
    with tf.name_scope(self.name + '/loss'):
      losses = tf.losses.mean_squared_error(labels=self.labels, predictions=self.outputs)
      self.loss = tf.reduce_mean(losses)
      tf.summary.histogram('losses', losses)
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope(self.name + '/train_op'):
      opt = tf.train.AdamOptimizer()
      self.train_op = opt.minimize(self.loss)

    with tf.name_scope(self.name + '/accuracy'):
      correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.outputs, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', self.accuracy)

    with tf.name_scope("global_step"):
      mf_global_step = tf.Variable(0, name='mf_global_step', trainable=False)
      self.mf_global_step_op = tf.assign_add(mf_global_step, 1)

      global_step = tf.Variable(0, name='global_step', trainable=False)
      self.global_step_op = tf.assign_add(global_step, 1)

  def predict(self, x):
    sess = K.get_session()
    assert self.outputs is not None, "Please set the outputs"
    return sess.run([self.outputs], feed_dict={self.inputs: x})

  def fit(self, x, y, mf_epochs=5, bp_epochs=10, use_nmf=True, use_backprop=False):
    """
    :param x: Train inputs.
    :param y: Train labels.
    :param mf_epochs: Matrix Factorization optimizer epochs
    :param bp_epochs: Back propagation epochs
    :param use_backprop: Whether use back propagation.
    """

    validation_split = 0.2
    train_size = len(x) * (1. - validation_split)
    mf_step_size = int(np.ceil(float(train_size * mf_epochs) / self.batch_size))
    bp_step_size = int(np.ceil(float(train_size * bp_epochs) / self.batch_size))

    init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(log_device_placement=self.use_gpu)) as sess:
      init.run()
      # TODO: should use tf.Session() but, if using that predict() cannot call.
      K.set_session(sess)
      # Summary
      summaries = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter('log', sess.graph)
      #
      assert self.inputs is not None, "Please set the First layer to self.inputs"

      # # TODO: Write more beautiful.
      if use_nmf:
        for epoch in range(mf_epochs):
          print("NMF Epoch {0} / {1}".format(epoch, mf_epochs))
          for step in range(mf_step_size):
            batch_x, batch_y = batch(x, y, batch_size=self.batch_size)
            feed_dict = {self.inputs: batch_x, self.labels: batch_y}
            self.optimizer.update(feed_dict=feed_dict, iter_size=self.iter_size)
            loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            # loss, acc, s, global_step = sess.run([self.loss, self.accuracy, summaries, self.mf_global_step_op],
            #                                      feed_dict=feed_dict)
            print("{0} / {1}: loss: {2} - acc: {3}".format(step * self.batch_size, train_size, loss, acc))
            # if summary_writer:
            #   summary_writer.add_summary(s, global_step)

      if use_backprop:
        for epoch in range(bp_epochs):
          print("Epoch {0} / {1}".format(epoch, bp_epochs))
          for step in range(bp_step_size):
            batch_x, batch_y = batch(x, y, batch_size=self.batch_size)
            feed_dict = {self.inputs: batch_x, self.labels: batch_y}
            _, loss, acc = sess.run(
                [self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
            # loss, acc, s, global_step = sess.run(
            #     [self.loss, self.accuracy, summaries, self.mf_global_step_op], feed_dict=feed_dict)
            print("{0} / {1}: loss: {2} - acc: {3}".format(step * self.batch_size, train_size, loss, acc))
            # if summary_writer:
            #   summary_writer.add_summary(s, global_step)


class DenseModel(Model):

  def __init__(self, **kwargs):
    super(DenseModel, self).__init__(**kwargs)

  def _build_model(self, batch_size=128):
    """
    Trainable Variable
    [<tf.Variable 'dense/kernel:0' shape=(784, 540) dtype=float32_ref>,
    <tf.Variable 'dense_1/kernel:0' shape=(540, 300) dtype=float32_ref>,
    <tf.Variable 'dense_2/kernel:0' shape=(300, 10) dtype=float32_ref>]
    """
    tf.reset_default_graph()
    print("model name:", self.name)
    with tf.name_scope(self.name + '/layers'):
      self.inputs = tf.placeholder(tf.float32, [batch_size, 784], name='inputs')
      self.labels = tf.placeholder(tf.float32, [batch_size, 10], name='labels')
      hidden1 = tf.layers.dense(
          self.inputs, 1000,
          use_bias=True, activation=tf.nn.relu)
      self.iter_size.append(3)
      self.hiddens.append(hidden1)
      hidden2 = tf.layers.dense(
          hidden1, 500,
          use_bias=True, activation=tf.nn.relu)
      print("Hidden2", hidden2)
      self.hiddens.append(hidden2)
      self.iter_size.append(3)
      self.outputs = tf.layers.dense(
          hidden2, 10,
          use_bias=True, activation=None)
      print("outputs", self.outputs)
      self.iter_size.append(1)


class Sin(Model):

  def __init__(self, **kwargs):
    super(Sin, self).__init__(**kwargs)

  def _build_model(self, batch_size=128):
    tf.reset_default_graph()
    with tf.name_scope(self.name):
      self.inputs = tf.placeholder(tf.float32, [batch_size, 1])
      self.labels = tf.placeholder(tf.float32, [batch_size, 1])
      self.outputs = tf.layers.dense(
          self.inputs, 1,
          use_bias=True, activation=tf.nn.sigmoid)


class BackpropModel(object):

  def __init__(self, *args, **kwargs):
    self._build_model()

  def _build_model(self, batch_size=128):
    """
    Trainable Variable
    [<tf.Variable 'dense/kernel:0' shape=(784, 540) dtype=float32_ref>,
    <tf.Variable 'dense_1/kernel:0' shape=(540, 300) dtype=float32_ref>,
    <tf.Variable 'dense_2/kernel:0' shape=(300, 10) dtype=float32_ref>]
    """
    tf.reset_default_graph()
    self.inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
    self.labels = tf.placeholder(tf.float32, [None, 10], name='labels')
    hidden1 = tf.layers.dense(
        self.inputs, 1000,
        use_bias=True, activation=tf.nn.relu)
    print("hidden1", hidden1)
    hidden2 = tf.layers.dense(
        hidden1, 500,
        use_bias=True, activation=tf.nn.relu)
    print("hidden2", hidden2)
    self.outputs = tf.layers.dense(
        hidden2, 10,
        use_bias=True, activation=None)

    losses = tf.losses.mean_squared_error(self.labels, self.outputs)
    self.loss = tf.reduce_mean(losses)
    opt = tf.train.AdamOptimizer()
    self.train_op = opt.minimize(self.loss)

    with tf.name_scope('accuracy') as scope:
      correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.outputs, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def train(self, x_train, y_train, validation_split=0.2, epochs=10, batch_size=100):
    train_size = len(x_train) * (1. - validation_split)
    step_size = int(np.ceil(float(train_size * epochs) / batch_size))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      init.run()
      for epoch in range(epochs):
        print("Epoch {0} / {1}".format(epoch, epochs))
        for step in range(step_size):
          batch_x, batch_y = batch(x_train, y_train, batch_size=batch_size)
          feed_dict = {self.inputs: batch_x, self.labels: batch_y}
          _, loss, acc = sess.run(
              [self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
          print("{0} / {1}: loss: {2} - acc: {3}".format(step * batch_size, train_size, loss, acc))
