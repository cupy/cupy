from cupy import core

gumbel_kernel = core.ElementwiseKernel(
    'T x, T loc, T scale', 'T y',
    'y = loc - log(-log(1 - x)) * scale',
    'gumbel_kernel'
)

laplace_kernel = core.ElementwiseKernel(
    'T x, T loc, T scale', 'T y',
    'y = (x < 0.5)? loc + scale * log(x + x):'
    ' loc - scale * log(2.0 - x - x)',
    'laplace_kernel'
)
