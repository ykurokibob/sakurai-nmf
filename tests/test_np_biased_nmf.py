import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import tensorflow as tf

from losses import np_frobenius_norm
from matrix_factorization.np_biased_nmf import _nonlin_solve, nonlin_semi_nmf, semi_nmf
from matrix_factorization.utility import relu

python_path = Path(__file__).parent
mat_file = python_path.joinpath('./large.mat').as_posix()

class TestNpBiasedNMF(tf.test.TestCase):
    def test_large_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        
        old_loss = np_frobenius_norm(a, u @ v)
        
        # a[1000, 5000] = u[1000, 501] x v[500, 5000]
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        u, v = semi_nmf(a, u, v)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        
        new_loss = np_frobenius_norm(a, u @ bias_v)
        
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('solve biased semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_solve_ax(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        v = _nonlin_solve(a=u, b=a, x=v, num_iters=2, solve_ax=True)
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        
        end_time = time.time()
        duration = end_time - start_time
        
        new_loss = np_frobenius_norm(a, u @ bias_v)
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('solve ax\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_solve_xa(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        old_loss = np_frobenius_norm(a, u @ v)
        
        start_time = time.time()
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        u = _nonlin_solve(a=v, b=a, x=u, num_iters=1, solve_ax=False)
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, u @ bias_v)
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('solve xa\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_large_nonlin_semi_nmf(self):
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        
        old_loss = np_frobenius_norm(a, u @ v)
        
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        u, v = nonlin_semi_nmf(a, u, v)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, u @ bias_v)
        
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('solve biased Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))
    
    def test_u_nega_nonlin_semi_nmf(self):
        mat_file = python_path.joinpath('./u_neg.mat').as_posix()
        auv = sio.loadmat(mat_file)
        a, u, v = auv['a'], auv['u'], auv['v']
        
        old_loss = np_frobenius_norm(a, relu(u @ v))
        
        u = np.hstack((u, np.ones((u.shape[0], 1))))
        start_time = time.time()
        
        u, v = nonlin_semi_nmf(a, u, v, num_calc_u=1, num_calc_v=1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        bias_v = np.vstack((v, np.ones((1, v.shape[1]))))
        new_loss = np_frobenius_norm(a, relu(u @ bias_v))
        
        assert new_loss < old_loss, "new loss should be less than old loss."
        print('solve u-nega biased Nonlinear semi-NMF\n\t'
              'old loss {0}\n\t'
              'new loss {1}\n\t'
              'process duration {2}'.format(old_loss, new_loss, duration))