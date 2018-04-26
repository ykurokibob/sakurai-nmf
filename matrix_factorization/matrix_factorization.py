import functools
import numpy as np
import tensorflow as tf


def semi_nmf(a, u, v,
             use_bias=False,
             use_tf=False,
             num_iters=1,
             rcond=1e-14,
             eps=1e-15,
             alpha=1e-2,
             beta=1e-2):
    """Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        use_bias: Use bias
        use_tf: When use Tensorflow, `a` should be instance of tf.placeholder
        num_iters: Number of iterations
        rcond: Reciprocal condition number
        eps:
        alpha: Coefficient for solve u.
        beta: Coefficient for solve v.

    Returns:
        When use TensorFlow, it returns operation u and v solved.
        When use NumPy, it returns results of u and v.
    """
    if use_bias:
        from matrix_factorization.np_biased_nmf import semi_nmf as semi_nmf_
        _semi_nmf = functools.partial(semi_nmf_,
                                      alpha=alpha,
                                      beta=beta,
                                      rcond=rcond,
                                      eps=eps,
                                      num_iters=num_iters,
                                      )
    else:
        from matrix_factorization.np_nmf import semi_nmf as semi_nmf_
        _semi_nmf = functools.partial(semi_nmf_,
                                      rcond=rcond,
                                      eps=eps,
                                      num_iters=num_iters,
                                      )
    if isinstance(a, np.ndarray) and not use_tf:
        return _semi_nmf(a=a, u=u, v=v)
    
    tf_u, tf_v = tf.py_func(_semi_nmf, [a, u, v], [tf.float64, tf.float64])
    
    return tf_u, tf_v


def nonlin_semi_nmf(a, u, v,
                    use_bias=False,
                    use_tf=False,
                    num_iters=1,
                    num_calc_u=1,
                    num_calc_v=1,
                    rcond=1e-14,
                    eps=1e-15,
                    alpha=1e-2,
                    beta=1e-2):
    """Nonlinear Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        use_bias: Use bias
        use_tf: When use Tensorflow, `a` should be instance of tf.placeholder
        num_iters: Number of iterations
        rcond: Reciprocal condition number
        num_calc_u: Number of calculating u.
        num_calc_v: Number of calculating v.
        eps:
        alpha: Coefficient for solve u.
        beta: Coefficient for solve v.

    Returns:

    """
    if use_bias:
        from matrix_factorization.np_biased_nmf import nonlin_semi_nmf as nonlin_semi_nmf_
        _nonlin_semi_nmf = functools.partial(nonlin_semi_nmf_,
                                             alpha=alpha,
                                             beta=beta,
                                             rcond=rcond,
                                             eps=eps,
                                             num_iters=num_iters,
                                             num_calc_u=num_calc_u,
                                             num_calc_v=num_calc_v,
                                             )
    else:
        from matrix_factorization.np_nmf import nonlin_semi_nmf as nonlin_semi_nmf_
        _nonlin_semi_nmf = functools.partial(nonlin_semi_nmf_,
                                             rcond=rcond,
                                             eps=eps,
                                             num_iters=num_iters,
                                             num_calc_u=num_calc_u,
                                             num_calc_v=num_calc_v,
                                             )
    
    if isinstance(a, np.ndarray) and not use_tf:
        return _nonlin_semi_nmf(a=a, u=u, v=v)

    tf_u, tf_v = tf.py_func(_nonlin_semi_nmf, [a, u, v], [tf.float64, tf.float64])

    return tf_u, tf_v
