"""Optimize neural network with Non-negative Matrix Factorization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matrix_factorization import nonlin_semi_nmf, semi_nmf

semi_nmf()
nonlin_semi_nmf()


class NMFOptimizer(object):
    
    def __init__(self):
        pass