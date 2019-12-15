import numpy

import cupy
from cupy import core


_preamble = '''
__device__ long long atomicAdd(long long *address, long long val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                     static_cast<unsigned long long>(val));
}'''

# TODO(unno): use searchsorted
_histogram_kernel = core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins',
    'raw U y',
    '''
    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    int high = n_bins - 1;
    int low = 0;

    while (high - low > 1) {
        int mid = (high + low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    atomicAdd(&y[low], U(1));
    ''',
    preamble=_preamble)


def histogram(x, bins=10):
    """Computes the histogram of a set of data.

    Args:
        x (cupy.ndarray): Input array.
        bins (int or cupy.ndarray): If ``bins`` is an int, it represents the
            number of bins. If ``bins`` is an :class:`~cupy.ndarray`, it
            represents a bin edges.

    Returns:
        tuple: ``(hist, bin_edges)`` where ``hist`` is a :class:`cupy.ndarray`
        storing the values of the histogram, and ``bin_edges`` is a
        :class:`cupy.ndarray` storing the bin edges.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram`
    """

    if x.dtype.kind == 'c':
        # TODO(unno): comparison between complex numbers is not implemented
        raise NotImplementedError('complex number is not supported')

    if isinstance(bins, int):
        if x.size == 0:
            min_value = 0.0
            max_value = 1.0
        else:
            min_value = float(x.min())
            max_value = float(x.max())
        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5
        bin_type = cupy.result_type(min_value, max_value, x)
        bins = cupy.linspace(min_value, max_value, bins + 1, dtype=bin_type)
    elif isinstance(bins, cupy.ndarray):
        if (bins[:-1] > bins[1:]).any():  # synchronize!
            raise ValueError('bins must increase monotonically.')
    else:
        raise NotImplementedError('Only int or ndarray are supported for bins')

    y = cupy.zeros(bins.size - 1, dtype='l')
    _histogram_kernel(x, bins, bins.size, y)
    return y, bins

# TODO(okuta): Implement histogram2d


# TODO(okuta): Implement histogramdd

_bincount_kernel = core.ElementwiseKernel(
    'S x', 'raw U bin',
    'atomicAdd(&bin[x], U(1))',
    'bincount_kernel',
    preamble=_preamble)
_bincount_with_weight_kernel = core.ElementwiseKernel(
    'S x, T w', 'raw U bin',
    'atomicAdd(&bin[x], w)',
    'bincount_with_weight_kernel')


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

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.bincount`

    """
    if x.ndim > 1:
        raise ValueError('object too deep for desired array')
    if x.ndim < 1:
        raise ValueError('object of too small depth for desired array')
    if x.dtype.kind == 'f':
        raise TypeError('x must be int array')
    if (x < 0).any():  # synchronize!
        raise ValueError('The first argument of bincount must be non-negative')
    if weights is not None and x.shape != weights.shape:
        raise ValueError('The weights and list don\'t have the same length.')
    if minlength is not None:
        minlength = int(minlength)
        if minlength < 0:
            raise ValueError('minlength must be non-negative')

    size = int(cupy.max(x)) + 1
    if minlength is not None:
        size = max(size, minlength)

    if weights is None:
        b = cupy.zeros((size,), dtype=numpy.intp)
        _bincount_kernel(x, b)
    else:
        b = cupy.zeros((size,), dtype=numpy.float64)
        _bincount_with_weight_kernel(x, weights, b)

    return b
