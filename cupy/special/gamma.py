import cupy
from cupy import core


_gamma_kernel = None


def _get_gamma_kernel():
    global _gamma_kernel
    if _gamma_kernel is None:
        _gamma_kernel = core.ElementwiseKernel(
            'T x', 'T y',
            'y = tgammaf(x)',
            'gamma_kernel'
        )
    return _gamma_kernel


def gamma(x):
    y = cupy.zeros_like(x)
    _get_gamma_kernel()(x, y)
    return y
