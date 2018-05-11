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
            y = tgammaf(x);
            """,
            'gamma_kernel'
        )
    return _gamma_kernel


def gamma(x):
    """Gamma function.

    .. seealso:: :data:`scipy.special.gamma`

    """
    x = cupy.asarray(x)
    if (x.dtype == cupy.float16 or x.dtype == cupy.dtype('b') or
            x.dtype == cupy.dtype('h') or x.dtype == cupy.dtype('B') or
            x.dtype == cupy.dtype('H') or x.dtype == cupy.bool_):
        x = x.astype(cupy.float32)
    elif (x.dtype == cupy.dtype('i') or x.dtype == cupy.dtype('l') or
            x.dtype == cupy.dtype('q') or x.dtype == cupy.dtype('I') or
            x.dtype == cupy.dtype('L') or x.dtype == cupy.dtype('Q')):
        x = x.astype(cupy.float64)
    y = cupy.zeros_like(x)
    _get_gamma_kernel()(x, y)
    return y
