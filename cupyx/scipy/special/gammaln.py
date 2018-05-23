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

    .. seealso:: :data:`scipy.special.gammaln`

    """
    if (x.dtype == cupy.float16 or x.dtype == cupy.dtype('b') or
            x.dtype == cupy.dtype('h') or x.dtype == cupy.dtype('B') or
            x.dtype == cupy.dtype('H') or x.dtype == cupy.bool_):
        x = x.astype(cupy.float32)
    elif (x.dtype == cupy.dtype('i') or x.dtype == cupy.dtype('l') or
            x.dtype == cupy.dtype('q') or x.dtype == cupy.dtype('I') or
            x.dtype == cupy.dtype('L') or x.dtype == cupy.dtype('Q')):
        x = x.astype(cupy.float64)
    y = cupy.zeros_like(x)
    _get_gammaln_kernel()(x, y)
    return y
