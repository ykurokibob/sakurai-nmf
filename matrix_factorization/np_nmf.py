import contextlib

import numpy as np


class AttrDict(dict):
    """Wrap a dictionary to access keys as attributes."""
    
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        super(AttrDict, self).__setattr__('_mutable', False)
    
    def __getattr__(self, key):
        # Do not provide None for unimplemented magic attributes.
        if key.startswith('__'):
            raise AttributeError
        return self.get(key, None)
    
    def __setattr__(self, key, value):
        if not self._mutable:
            message = "Cannot set attribute '{}'.".format(key)
            message += " Use 'with obj.unlocked:' scope to set attributes."
            raise RuntimeError(message)
        if key.startswith('__'):
            raise AttributeError("Cannot set magic attribute '{}'".format(key))
        self[key] = value
    
    @property
    @contextlib.contextmanager
    def unlocked(self):
        super(AttrDict, self).__setattr__('_mutable', True)
        yield
        super(AttrDict, self).__setattr__('_mutable', False)
    
    def copy(self):
        return type(self)(super(AttrDict, self).copy())


def relu(x):
    return x * (x > 0)


def low_rank(a, rcond=1e-10):
    def _makearray(a):
        new = np.asarray(a)
        wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
        return new, wrap
    
    def _isEmpty2d(arr):
        # check size first for efficiency
        return arr.size == 0 and np.product(arr.shape[-2:]) == 0
    
    a, wrap = _makearray(a)
    rcond = np.asarray(rcond)
    if _isEmpty2d(a):
        res = np.empty(a.shape[:-2] + (a.shape[-1], a.shape[-2]), dtype=a.dtype)
        return wrap(res)
    a = a.conjugate()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    
    # discard small singular values
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0
    
    return AttrDict(u=u, s=np.diag(s), vt=vt)


def _low_rank(a, rcond=1e-14):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    v = np.transpose(vt)
    _s = np.diag(s)
    _s = _s / np.max(_s)
    k = np.sum(_s > rcond)
    
    u = u[:, :k]
    s = np.diag(s)[:k, :k]
    v = v[:, :k]
    return AttrDict(u=u, s=s, v=v)


def semi_nmf(a, u, v, rcond=1e-14, eps=1e-15):
    svd = _low_rank(v, rcond=rcond)
    u_t = np.transpose(svd.u)
    # _v = np.transpose(svd.vt)
    _v = svd.v
    s_inv = np.linalg.inv(svd.s)
    u = ((a @ _v) @ s_inv) @ u_t
    # v_inv =((a @ svd.vt) / svd.s) * u_t
    
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


def nonlin_solve(a, b, x, num_iters, solve_ax=True):
    """
    :param a:
    :param b:
    :param x:
    :param num_iters:
    :return: x
    """
    a_svd = _low_rank(a)
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


def nonlin_semi_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1):
    for _ in range(num_iters):
        u = nonlin_solve(v, a, u, 1, solve_ax=False)
        v = nonlin_solve(u, a, v, 1, solve_ax=True)
    return u, v