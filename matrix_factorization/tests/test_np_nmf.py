import scipy.io as sio

import time
from losses import np_frobenius_norm
from matrix_factorization.np_nmf import *


def test_low_rank():
    a = np.random.uniform(0., 1., size=[5, 5])
    svd = low_rank(a)
    u, s, vt = svd.u, svd.s, svd.vt
    np.testing.assert_equal((u @ s @ vt).shape, a.shape)


def test_semi_nmf():
    auv = sio.loadmat('auv.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    u, v = semi_nmf(a, u, v)
    loss = np_frobenius_norm(a, u @ v)
    
    assert a.shape == (u @ v).shape
    print('loss {0}'.format(loss))


def test_large_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    u, v = semi_nmf(a, u, v)
    loss = np_frobenius_norm(a, u @ v)
    
    assert a.shape == (u @ v).shape
    print('loss {0}'.format(loss))


def test_solve_ax():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    v = nonlin_solve(a=u, b=a, x=v, num_iters=2, solve_ax=True)
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, u @ v)
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve ax\n\told loss {0}\n\tnew loss {1}\n\tprocess duration {2}'.format(old_loss, new_loss, duration))


def test_solve_xa():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    u = nonlin_solve(a=v, b=a, x=u, num_iters=1, solve_ax=False)
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, u @ v)
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve xa\n\told loss {0}\n\tnew loss {1}\n\tprocess duration {2}'.format(old_loss, new_loss, duration))


def test_large_nonlin_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    old_loss = np_frobenius_norm(a, u @ v)
    start_time = time.time()
    u, v = nonlin_semi_nmf(a, u, v)
    end_time = time.time()
    duration = end_time - start_time
    new_loss = np_frobenius_norm(a, u @ v)
    
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve nonlinear semi-NMF\n\told loss {0}\n\tnew loss {1}\n\tprocess duration {2}'.format(old_loss, new_loss, duration))