import numpy as np
import tensorflow as tf

from simple_model.dataset import load_data


def tround(x, digits=3):
  return tf.round(x * 10 ** digits)


class Sin(object):

  def __init__(self):
    self._build_model()

  def _build_model(self):
    tf.reset_default_graph()
    with tf.name_scope('models'):
      self.x = tf.placeholder(tf.float32, [None, 1])
      self.y = tf.placeholder(tf.float32, [None, 1])
      self.outputs = tf.layers.dense(self.x, 1)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tround(self.y, 2), tround(self.outputs, 2))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.
      tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('loss'):
      losses = tf.losses.mean_squared_error(labels=self.y, predictions=self.outputs)
      self.loss = tf.reduce_mean(losses)
      tf.summary.histogram('losses', losses)
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('train_op'):
      opt = tf.train.AdamOptimizer()
      self.train_op = opt.minimize(self.loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    self.global_step_op = tf.assign_add(global_step, 1)

  def predict(self, sess: tf.Session, x):
    return sess.run(self.outputs, feed_dict={self.x, x})

  def fit(self, sess: tf.Session, x, y, batch_size=100):
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log', sess.graph)

    epoch_num = 100
    step_size = int(np.ceil(float(len(x) * epoch_num) / batch_size))
    for i in range(1, epoch_num + 1):
      for step in range(1 * step_size + 1):
        random_index = np.random.choice(len(x), batch_size)
        batch_x = x[random_index]
        batch_y = y[random_index]
        s, _, loss, global_step = sess.run([summaries, self.train_op, self.loss, self.global_step_op],
                                           feed_dict={self.x: batch_x, self.y: batch_y})
        print('loss: ', loss)
        if summary_writer:
          summary_writer.add_summary(s, global_step)


def main(_):
  (x_train, y_train) = load_data()
  model = Sin()
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    init.run()
    model.fit(sess, x_train, y_train)


if __name__ == '__main__':
  tf.app.run()
