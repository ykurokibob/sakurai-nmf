import time

import numpy as np
import scipy.io as sio

from losses import np_frobenius_norm
from matrix_factorization.np_biased_nmf import semi_nmf


def test_large_semi_nmf():
    auv = sio.loadmat('large.mat')
    a, u, v = auv['a'], auv['u'], auv['v']
    
    old_loss = np_frobenius_norm(a, u @ v)
    
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