from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np


def main():
    num_iterts = 10
    _size = [1000, 1000]
    _inputs = np.random.uniform(0., 1., size=_size)
    svd = np.linalg.svd
    
    start_time = time.time()
    
    for _ in range(num_iterts):
        svd(_inputs)
    
    end_time = time.time()
    
    duration = (end_time - start_time) / num_iterts
    print('Duration of computing svd: {}.'.format(duration))


if __name__ == '__main__':
    np.random.seed(42)
    main()