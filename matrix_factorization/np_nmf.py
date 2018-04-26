"""semi-NMF and Nonlinear semi-NMF solver by NumPy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from matrix_factorization.utility import _low_rank, relu


def semi_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1):
    """Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:

    Returns:
        u, v
    """
    for _ in range(num_iters):
        svd = _low_rank(v, rcond=rcond)
        u_t = np.transpose(svd.u)
        _v = svd.v
        s_inv = np.linalg.inv(svd.s)
        u = ((a @ _v) @ s_inv) @ u_t
        
        u_t = np.transpose(u)
        uta = u_t @ a
        u_ta_p = (np.abs(uta) + uta) * 0.5
        u_ta_m = (np.abs(uta) - uta) * 0.5
        utu = u_t @ u
        u_tu_p = (np.abs(utu) + utu) * 0.5
        u_tu_m = (np.abs(utu) - utu) * 0.5
        
        uvm = u_tu_m @ v
        uvp = u_tu_p @ v
        sqrt = np.sqrt(
            np.divide(u_ta_p + uvm,
                      u_ta_m + uvp + eps)
        )
        v = np.multiply(v, sqrt)
    
    return u, v


def _nonlin_solve(a, b, x, rcond=1e-14, num_iters=1, solve_ax=True):
    """Nonlinear Solver.
    Args:
        num_iters: Number of iterations each solving.
        solve_ax: Whether to solve min_x || b - f(ax) || or min_x || b - f(xa) ||
    """
    a_svd = _low_rank(a, rcond=rcond)
    u = a_svd.u
    s = a_svd.s
    v = a_svd.v
    
    _omega = 1.0
    
    def _solve_ax(x):
        """
         min_x || b - f(ax) ||
        """
        for _ in range(num_iters):
            r = b - relu(a @ x)
            ur = u.T @ r
            solve = np.linalg.solve(s, ur)
            x = x + _omega * (v @ solve)
        return x
    
    def _solve_xa(x):
        """
         min_x || b - f(xa) ||
        """
        for _ in range(num_iters):
            r = b - relu(x @ a)
            s_inv = np.linalg.inv(s)
            rvs = r @ v @ s_inv
            x = x + _omega * (rvs @ u.T)
        return x
    
    if solve_ax:
        return _solve_ax(x)
    else:
        return _solve_xa(x)


def nonlin_semi_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1, num_calc_u=1, num_calc_v=1):
    """Nonlinear semi-NMF
    
    Args:
        a: Original non-negative matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        num_iters: number of iterations

    Returns:
        Solved u, v
    """
    for _ in range(num_iters):
        u = _nonlin_solve(v, a, u, rcond=rcond, num_iters=num_calc_u, solve_ax=False)
        v = _nonlin_solve(u, a, v, rcond=rcond, num_iters=num_calc_v, solve_ax=True)
    return u, v