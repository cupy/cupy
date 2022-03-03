import math

import cupy
from cupy import _core


gammaln = _core.create_ufunc(
    'cupyx_scipy_special_gammaln', ('f->f', 'd->d'),
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


def multigammaln(a, d):
    r"""Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : cupy.ndarray
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space of integration.

    Returns
    -------
    res : ndarray
        The values of the log multivariate gamma at the given points `a`.

    See Also
    --------
    :func:`scipy.special.multigammaln`

    """
    if not cupy.isscalar(d) or (math.floor(d) != d):
        raise ValueError("d should be a positive integer (dimension)")
    if cupy.isscalar(a):
        a = cupy.asarray(a, dtype=float)
    if int(cupy.any(a <= 0.5 * (d - 1))):
        raise ValueError("condition a > 0.5 * (d-1) not met")
    res = (d * (d - 1) * 0.25) * math.log(math.pi)
    gam0 = gammaln(a)
    if a.dtype.kind != 'f':
        # make sure all integer dtypes do the summation with float64
        gam0 = gam0.astype(cupy.float64)
    res = res + gam0
    for j in range(2, d + 1):
        res += gammaln(a - (j - 1.0) / 2)
    return res
