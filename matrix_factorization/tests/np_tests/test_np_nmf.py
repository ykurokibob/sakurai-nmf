import time

import scipy.io as sio

from losses import np_frobenius_norm
from matrix_factorization.np_nmf import _nonlin_solve, nonlin_semi_nmf, semi_nmf
from matrix_factorization.utility import relu


def test_semi_nmf():
    auv = sio.loadmat('auv.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    
    u, v = semi_nmf(a, u, v)
    
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, u @ v)
    assert a.shape == (u @ v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_large_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    
    u, v = semi_nmf(a, u, v)
    
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, u @ v)
    assert a.shape == (u @ v).shape
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve (large) semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_solve_ax():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    v = _nonlin_solve(a=u, b=a, x=v, num_iters=2, solve_ax=True)
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, relu(u @ v))
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve ax\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_solve_xa():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    old_loss = np_frobenius_norm(a, u @ v)
    
    start_time = time.time()
    u = _nonlin_solve(a=v, b=a, x=u, num_iters=1, solve_ax=False)
    end_time = time.time()
    duration = end_time - start_time
    
    new_loss = np_frobenius_norm(a, relu(u @ v))
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve xa\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))


def test_large_nonlin_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    old_loss = np_frobenius_norm(a, u @ v)
    start_time = time.time()
    u, v = nonlin_semi_nmf(a, u, v)
    end_time = time.time()
    duration = end_time - start_time
    new_loss = np_frobenius_norm(a, relu(u @ v))
    
    assert new_loss < old_loss, "new loss should be less than old loss."
    print('solve nonlinear semi-NMF\n\t'
          'old loss {0}\n\t'
          'new loss {1}\n\t'
          'process duration {2}'.format(old_loss, new_loss, duration))