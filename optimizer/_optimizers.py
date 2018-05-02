from collections import namedtuple

from matrix_factorization import *

Layer = namedtuple('Layer', ['kernel', 'use_bias'])
Updater = namedtuple('Updater', ['u_op', 'v_op', 'loss', 'update_kernel', 'update_v'])


def get_name(x):
  if x is None:
    return None
  return x.name.split('/')[0]


class Optimizer(object):

  def __init__(self,
               max_timesteps=1,
               mf_iteration=5,
               epsilon=1e-5,
               name=None):
    self.weights = []
    self.matrix_facs = []
    self.hidden_z = None
    self.updater = []

    self.mf_iteration = mf_iteration
    self.epsilon = epsilon
    self.name = name or self.__class__.__name__
    self.sess = None
    self.loss_func = None
    self.max_timesteps = max_timesteps

  # TODO: 2. Back propagation to update weights.
  def update(self, feed_dict, iter_size):
    assert len(feed_dict) == 2, "feed_dict have to get x_ph and labels"
    self.sess = K.get_session()
    inv_iter = list(reversed(iter_size))

    # TODO from Arai: Enable to specify the numbers of iteration for factorize U and V.
    # Each Layer Factorization.
    for i, (update, factorize_iter) in enumerate(zip(self.updater, inv_iter)):
      print("Update_V", update.update_v)
      # print(i, factorize_iter, update)
      for j in range(factorize_iter):
        # Compute u
        for _ in range(self.matrix_facs[i].u_iter):
          self.sess.run(update.u_op, feed_dict=feed_dict)
        # Compute v
        for _ in range(self.matrix_facs[i].v_iter):
          self.sess.run(update.v_op, feed_dict=feed_dict)
        # Update U.
        _ = self.sess.run([update.update_kernel], feed_dict=feed_dict)
        loss = self.sess.run(update.loss, feed_dict=feed_dict)
        print("[{0}:{1}] - loss {2}".format(self.matrix_facs[i].name, j, loss))

  def init_matrix(self, labels, hiddens):
    self.hidden_z = hiddens  # For update init V.
    self._set_kernels()
    reversed_weight = list(reversed(self.weights))
    reversed_hidden = list(reversed(self.hidden_z))
    target_y = labels
    for i, (layer, hidden) in enumerate(zip(reversed_weight, reversed_hidden)):
      target_y = self._factorize_process(i, layer, target_y, hidden)

  def _set_kernels(self):
    weights = tf.trainable_variables() + [None]
    for i, weight in enumerate(weights):
      if weight is None:
        break
      if "kernel" in weight.name:
        if get_name(weights[i + 1]) == get_name(weight):
          use_bias = True
        else:
          use_bias = False
        self.weights.append(Layer(weight, use_bias))

    # Validation.
    shape = K.get_variable_shape(self.weights[0].kernel)
    for i, layer in enumerate(self.weights[1:], start=1):
      next_shape = K.get_variable_shape(layer.kernel)
      assert shape[1] == next_shape[0], "not valid shape models"
      shape = next_shape
    print("valid models...")

  def _factorize_process(self, i, layer, z, hidden_z):
    raise NotImplementedError


class NMF(Optimizer):
  def __init__(self, **kwargs):
    super(NMF, self).__init__(**kwargs)

  def _factorize_process(self, i, layer, z, hidden_z):
    u_shape = K.get_variable_shape(layer.kernel)
    v_shape = (K.get_variable_shape(z)[0], u_shape[0])
    is_first = (i == len(self.weights))

    # Last Layer
    if i == 0:
      mf = SemiNMF(z, v_shape, u_shape, name=i, use_bias=layer.use_bias)
    # Other layers.
    else:
      mf = NonlinearSemiNMF(z, v_shape, u_shape, is_first=is_first, name=i, use_bias=layer.use_bias)

    u_op, v_op, _, mf_loss = mf.factorize()
    self.matrix_facs.append(mf)
    update_kernel = K.update(layer.kernel, mf.algorithm.u)

    # TODO from Arai: V's value is Last TargetY.
    if len(self.weights) - 1 > i:
      update_v = tf.assign(self.matrix_facs[i].algorithm.v,
                           hidden_z, name="update_v/{0}".format(i))
    else:
      update_v = tf.no_op(name="update_v/First")

    self.updater.append(Updater(u_op=u_op,
                                v_op=v_op,
                                loss=mf_loss,
                                update_kernel=update_kernel,
                                update_v=update_v))

    return mf.algorithm.v
