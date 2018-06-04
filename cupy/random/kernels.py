from cupy import core

_gumbel_kernel = None
_laplace_kernel = None


def _get_gumbel_kernel():
    global _gumbel_kernel
    if _gumbel_kernel is None:
        _gumbel_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = loc - log(-log(1 - x)) * scale',
            'gumbel_kernel'
        )
    return _gumbel_kernel


def _get_laplace_kernel():
    global _laplace_kernel
    if _laplace_kernel is None:
        _laplace_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = (x < 0.5)? loc + scale * log(x + x):'
            ' loc - scale * log(2.0 - x - x)',
            'laplace_kernel'
        )
    return _laplace_kernel
