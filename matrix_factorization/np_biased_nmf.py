import numpy as np

from matrix_factorization.utility import _low_rank


def semi_nmf(a, u, v, alpha=1e-2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1):
    """Biased Semi-NMF
    Args:
        a: original matrix factorized
        u: left matrix
        v: non-negative matrix
        rcond:
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
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        bias_v = np.vstack((v, bias.T))
    return u, v


def nonlin_semi_nmf(a, u, v, alpha=1e2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1):
    """Biased Nonlinear Semi-NMF
    Args:
        a:
        u:
        v:
        alpha:
        beta:
        rcond:
        eps:
        num_iters:

    Returns:

    """
    n = v.shape[1]
    bias = np.ones((n, 1))
    bias_v = np.vstack((v, bias.T))
    ie = np.ones((n, n))
    return bias_v, ie
    # for _ in range(num_iters):
    #     pass