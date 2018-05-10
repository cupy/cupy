import cupy
from cupy import core


_gammaln_kernel = None


def _get_gammaln_kernel():
    global _gammaln_kernel
    if _gammaln_kernel is None:
        _gammaln_kernel = core.ElementwiseKernel(
            'T x', 'T y',
            'y = lgammaf(x)',
            'gammaln_kernel'
        )
    return _gammaln_kernel


def gammaln(x):
    y = cupy.zeros_like(x)
    _get_gammaln_kernel()(x, y)
    return y
