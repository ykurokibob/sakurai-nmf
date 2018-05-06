import numpy as np
import scipy.io as sio


def large_ua_neg():
    a = np.random.uniform(-1000., 1000., size=[5000, 1500])
    u = np.random.uniform(0., 1000., size=[5000, 3000])
    v = np.random.uniform(-1000., 1000., size=(3000, 1500))
    assert a.shape == (u @ v).shape
    sio.savemat('large_ua_neg.mat', dict(a=a, u=u, v=v))


def small_v_neg():
    a = np.random.uniform(-1000., 1000., size=[500, 1000]).astype('float64')
    u = np.random.uniform(0., 1000., size=[500, 700]).astype('float64')
    v = np.random.uniform(-1000., 1000., size=[700, 1000]).astype('float64')
    assert a.shape == (u @ v).shape
    sio.savemat('small_v_neg.mat', dict(a=a, u=u, v=v))
    
def large_v_neg():
    a = np.random.uniform(-1000., 1000., size=[5000, 3000]).astype('float64')
    u = np.random.uniform(0., 1000., size=[5000, 6000]).astype('float64')
    v = np.random.uniform(-1000., 1000., size=[6000, 3000]).astype('float64')
    assert a.shape == (u @ v).shape
    sio.savemat('large_v_neg.mat', dict(a=a, u=u, v=v))
    


def large_tf_format():
    a = np.random.uniform(-1., 1., size=[5000, 1500])
    u = np.random.uniform(0., 1., size=[5000, 300])
    v = np.random.uniform(0., 1., size=(300, 1500))
    assert a.shape == (u @ v).shape
    sio.savemat('large_u_neg_tf_format.mat', dict(a=a, u=u, v=v))


def main():
    # large_ua_neg()
    # large_tf_format()
    # small_v_neg()
    large_v_neg()


if __name__ == '__main__':
    main()