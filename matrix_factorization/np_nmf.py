import numpy as np


def relu(x):
  return x * (x > 0)


def nonlinear_snmf(init_u, init_v, target_y, f=relu, epsilon=1e-5, iteration=1000000, freq_iter=100):
  u = init_u
  v = init_v
  y = target_y
  err = 0
  for k in range(0, iteration):
    v_inv = np.linalg.pinv(v)
    u = u + (y - f(u @ v)) @ v_inv
    u_inv = np.linalg.pinv(u)
    v = f(v + u_inv @ (y - f(u @ v)))

    # Early Stopping.
    err_mat = y - f(u @ v)
    err = np.linalg.norm(err_mat)
    if err < epsilon:
      np.testing.assert_almost_equal(y, f(u @ v))
      break

    if k % freq_iter == 0:
      print('loss:', err)
  return u, v, err


def main():
  long_s = 784
  short_s = 700
  U = np.random.randn(*(long_s, short_s))
  V = np.random.uniform(0, 1, size=(short_s, long_s))
  target = np.random.uniform(0, 3, size=(long_s, long_s))
  u, v, err = nonlinear_snmf(U, V, target, freq_iter=10)
  print("Error: {0}".format(err))
  print("u={0}\n\nv={1}".format(u, v))


if __name__ == '__main__':
  main()
