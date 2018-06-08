import cupy
from cupy import core


_gammaln_kernel = None


def _get_gammaln_kernel():
    global _gammaln_kernel
    if _gammaln_kernel is None:
        _gammaln_kernel = core.ElementwiseKernel(
            'T x', 'T y',
            """
            if(isinf(x) && x < 0){
                y = - 1.0 / 0.0;
                return;
            }
            y = lgamma(x);
            """,
            'gammaln_kernel'
        )
    return _gammaln_kernel


def gammaln(x):
    """Logarithm of the absolute value of the Gamma function.

    Args:
        x (cupy.ndarray): Values on the real line at which to compute
        ``gammaln``.

    Returns:
        cupy.ndarray: Values of ``gammaln`` at x.

    .. seealso:: :data:`scipy.special.gammaln`

    """
    if x.dtype.char in '?ebBhH':
        x = x.astype(cupy.float32)
    elif x.dtype.char in 'iIlLqQ':
        x = x.astype(cupy.float64)
    y = cupy.zeros_like(x)
    _get_gammaln_kernel()(x, y)
    return y
