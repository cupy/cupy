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


def gamma(z):
    """Gamma function.

    Args:
        z (cupy.ndarray): The input of gamma function.

    Returns:
        cupy.ndarray: Computed value of gamma function.

    .. seealso:: :data:`scipy.special.gamma`

    """
    if z.dtype.char in '?ebBhH':
        z = z.astype(cupy.float32)
    elif z.dtype.char in 'iIlLqQ':
        z = z.astype(cupy.float64)
    y = cupy.zeros_like(z)
    _get_gamma_kernel()(z, y)
    return y
