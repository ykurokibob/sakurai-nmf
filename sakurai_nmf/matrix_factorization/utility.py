"""Utility for construct algorithms"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def _low_rank(a, rcond=1e-14):
    # TODO
    assert not np.isnan(a).any(), have_nan('a', a)
    u, s, vt = np.linalg.svd(a, full_matrices=True)
    v = np.transpose(vt)
    _s = np.diag(s)
    _s = _s / np.max(_s)
    k = np.sum(_s > rcond)
    
    u = u[:, :k]
    s = np.diag(s)[:k, :k]
    v = v[:, :k]
    assert not np.isnan(u).any(), have_nan('u', u)
    assert not np.isnan(s).any(), have_nan('s', s)
    assert not np.isnan(v).any(), have_nan('v', v)
    return AttrDict(u=u, s=s, v=v)


def have_nan(name, matrix: np.ndarray):
    vec = matrix.flatten()
    num_nans = np.isnan(vec).sum()
    message = '{} has Nan the number of {}'.format(name, num_nans)
    return message


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference