"""Biased semi-NMF and Nonlinear semi-NMF solver by NumPy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from matrix_factorization.utility import _low_rank, relu


def semi_nmf(a, u, v, alpha=1e-2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1):
    """Biased Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:

    Returns:
        u, v
    """
    n = v.shape[1]
    bias = np.ones((n, 1))
    bias_v = np.vstack((v, bias.T))
    # u = np.hstack((u, np.ones((u.shape[0], 1))))
    
    for _ in range(num_iters):
        svd = _low_rank(bias_v, rcond=rcond)
        u_t = np.transpose(svd.u)
        r = a - u @ bias_v
        rv = r @ svd.v
        s_inv = np.linalg.inv(svd.s)
        u = u + (rv @ s_inv) @ u_t
        ss = np.diag(svd.s)
        ss_square = np.square(ss)
        ss = np.divide(ss_square,
                       (alpha + ss_square))
        u = u @ (svd.u @ np.diag(ss) @ u_t)
        
        u_org = u[:, :-1]
        u_t = np.transpose(u_org)
        ua = u_t @ a
        uap = (np.abs(ua) + ua) * 0.5
        uam = (np.abs(ua) - ua) * 0.5
        uu = u_t @ u
        uup = (np.abs(uu) + uu) * 0.5
        uum = (np.abs(uu) - uu) * 0.5
        
        divide = np.divide(uap + uum @ bias_v + beta * v,
                           uam + uup @ bias_v + beta * v + eps)
        # TODO: The divide induce Nan.
        divide[divide < 0.] = 0.
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        bias_v = np.vstack((v, bias.T))
    return u, v


def _nonlin_solve(a, b, x, _lambda=1e-2, rcond=1e-14, eps=1e-15, num_iters=1, solve_ax=True):
    """Nonlinear Solver.
    Args:
        num_iters: Number of iterations each solving.
        solve_ax: Whether to solve min_x || b - f(ax) || or min_x || b - f(xa) ||
    """
    _omega = 1.0
    
    def _solve_ax(x):
        """
         min_x || b - f(ax) ||
        """
        a_svd = _low_rank(a[:, :-1], rcond=1e-14)
        u = a_svd.u
        s = a_svd.s
        v = a_svd.v
        
        n = a.shape[1]
        bias = np.ones((x.shape[1], 1))
        bias_x = np.vstack((x, bias.T))
        _aa = a[:, :-1].T @ a[:, :-1]
        u_svd = _low_rank(_aa, rcond=rcond)
        
        for _ in range(num_iters):
            r = b - relu(a @ bias_x)
            ur = u.T @ r
            sur_solve = np.linalg.solve(s, ur)
            x = x + _omega * (v @ sur_solve)
            _eye = np.eye(n - 1)
            su_solve = np.linalg.solve(u_svd.s, u_svd.u.T)
            vsu = u_svd.v @ su_solve
            _x = _eye + _lambda * vsu
            x = np.linalg.solve(_x, x)
            bias_x = np.vstack((x, bias.T))
        return x
    
    def _solve_xa(x):
        """
         min_x || b - f(xa) ||
        """
        bias = np.ones((a.shape[1], 1))
        _a = np.vstack((a, bias.T))
        a_svd = _low_rank(_a, rcond=rcond)
        u = a_svd.u
        s = a_svd.s
        v = a_svd.v
        
        bias = np.ones((a.shape[1], 1))
        bias_x = np.vstack((a, bias.T))
        for _ in range(num_iters):
            r = b - relu(x @ bias_x)
            rv = r @ v
            s_inv = np.linalg.inv(s)
            rvs = rv @ s_inv
            x = x + _omega * (rvs @ u.T)
            ss = np.diag(s)
            ss = np.divide(
                np.square(ss),
                np.square(ss) + _lambda)
            x = x @ (u @ np.diag(ss) @ u.T)
        return x
    
    if solve_ax:
        return _solve_ax(x)
    else:
        return _solve_xa(x)


def nonlin_semi_nmf(a, u, v, alpha=1e2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1, num_calc_u=1, num_calc_v=1):
    """Biased Nonlinear Semi-NMF
    Args:
        a: Original non-negative matrix factorized
        u: Biased-matrix
        v: Non-negative matrix
        alpha: Coefficient for solve u.
        beta: Coefficient for solve v.
        rcond: Reciprocal condition number
        eps:
        num_iters: Number of iterations

    Returns:

    """
    for _ in range(num_iters):
        u = _nonlin_solve(v, a, u, _lambda=alpha, rcond=rcond, eps=eps, solve_ax=False, num_iters=num_calc_u)
        v = _nonlin_solve(u, a, v, _lambda=beta, rcond=rcond, eps=eps, solve_ax=True, num_iters=num_calc_v)
    return u, v