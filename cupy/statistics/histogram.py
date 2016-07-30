import numpy

import cupy


# TODO(okuta): Implement histogram


# TODO(okuta): Implement histogram2d


# TODO(okuta): Implement histogramdd


def bincount(x, weights=None, minlength=None):
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        x (cupy.ndarray): Input array.
        weights (cupy.ndarray): Weights array which has the same shape as
            ``x``.
        minlength (int): A minimum number of bins for the output array.

    Returns:
        cupy.ndarray: The result of binning the input array. The length of
            output is equal to ``max(cupy.max(x) + 1, minlength)``.

    .. seealso:: :func:`numpy.bincount`

    """
    if x.ndim > 1:
        raise ValueError('object too deep for desired array')
    if x.ndim < 1:
        raise ValueError('object of too small depth for desired array')
    if x.dtype.kind == 'f':
        raise TypeError('x must be int array')
    if (x < 0).any():
        raise ValueError('The first argument of bincount must be non-negative')
    if weights is not None and x.shape != weights.shape:
        raise ValueError('The weights and list don\'t have the same length.')
    if minlength is not None:
        minlength = int(minlength)
        if minlength <= 0:
            raise ValueError('minlength must be positive')

    size = int(cupy.max(x)) + 1
    if minlength is not None:
        size = max(size, minlength)

    if weights is None:
        # atomicAdd for int64 is not provided
        b = cupy.zeros((size,), dtype=cupy.int32)
        cupy.ElementwiseKernel(
            'S x', 'raw U bin',
            'atomicAdd(&bin[x], 1)',
            'bincount_kernel'
        )(x, b)
        b = b.astype(numpy.intp)
    else:
        # atomicAdd for float64 is not provided
        b = cupy.zeros((size,), dtype=cupy.float32)
        cupy.ElementwiseKernel(
            'S x, T w', 'raw U bin',
            'atomicAdd(&bin[x], w)',
            'bincount_with_weight_kernel'
        )(x, weights, b)
        b = b.astype(cupy.float64)

    return b


# TODO(okuta): Implement digitize
