from cupy import core


gammaln = core.create_ufunc(
    'cupyx_scipy_gammaln', ('f->f', 'd->d'),
    '''
    if (isinf(in0) && in0 < 0) {
        out0 = -1.0 / 0.0;
    } else {
        out0 = lgamma(in0);
    }
    ''',
    doc="""Logarithm of the absolute value of the Gamma function.

    Args:
        x (cupy.ndarray): Values on the real line at which to compute
        ``gammaln``.

    Returns:
        cupy.ndarray: Values of ``gammaln`` at x.

    .. seealso:: :data:`scipy.special.gammaln`

    """)
