import cupy
from cupy import core


_gamma_kernel = None


def _get_gamma_kernel():
    global _gamma_kernel
    if _gamma_kernel is None:
        _gamma_kernel = core.ElementwiseKernel(
            'T x', 'T y',
            """
            if(isinf(x) && x < 0){
                y = - 1.0 / 0.0;
                return;
            }
            if(x < 0. && x == floor(x)){
                y = 1.0 / 0.0;
                return;
            }
            y = tgamma(x);
            """,
            'gamma_kernel'
        )
    return _gamma_kernel


def gamma(x):
    """Gamma function.

    .. seealso:: :data:`scipy.special.gamma`

    """
    if x.dtype.char in '?ebBhH':
        x = x.astype(cupy.float32)
    elif x.dtype.char in 'iIlLqQ':
        x = x.astype(cupy.float64)
    y = cupy.zeros_like(x)
    _get_gamma_kernel()(x, y)
    return y
