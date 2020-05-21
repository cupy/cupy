import operator
import warnings

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


_weighted_histogram_kernel = core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins, raw W weights',
    'raw Y y',
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
    atomicAdd(&y[low], (Y)weights[i]);
    ''',
    preamble=_preamble)


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == cupy.bool_:
        warnings.warn("Converting input from {} to {} for compatibility."
                      .format(a.dtype, cupy.uint8),
                      RuntimeWarning, stacklevel=3)
        a = a.astype(cupy.uint8)

    if weights is not None:
        if not isinstance(weights, cupy.ndarray):
            raise ValueError("weights must be a cupy.ndarray")
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (numpy.isfinite(first_edge) and numpy.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(
                    first_edge, last_edge))
    elif a.size == 0:
        first_edge = 0.0
        last_edge = 1.0
    else:
        first_edge = float(a.min())
        last_edge = float(a.max())
        if not (cupy.isfinite(first_edge) and cupy.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(
                    first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range):
    """
    Computes the bins used internally by `histogram`.

    Args:
        a (ndarray): Ravelled data array
        bins (int or ndarray): Forwarded argument from `histogram`.
        range (None or tuple): Forwarded argument from `histogram`.

    Returns:
        bin_edges (ndarray): Array of bin edges
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, int):  # cupy.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError:
            raise TypeError(
                '`bins` must be an integer, a string, or an array')
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges(a, range)

    elif isinstance(bins, cupy.ndarray):
        if bins.ndim == 1:    # cupy.ndim(bins) == 0:
            bin_edges = cupy.asarray(bins)
            if (bin_edges[:-1] > bin_edges[1:]).any():  # synchronize!
                raise ValueError(
                    '`bins` must increase monotonically, when an array')

    elif isinstance(bins, str):
        raise NotImplementedError(
            "only integer and array bins are implemented")

    if n_equal_bins is not None:
        # numpy's gh-10322 means that type resolution rules are dependent on
        # array shapes. To avoid this causing problems, we pick a type now and
        # stick with it throughout.
        bin_type = cupy.result_type(first_edge, last_edge, a)
        if cupy.issubdtype(bin_type, cupy.integer):
            bin_type = cupy.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = cupy.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type)
    return bin_edges


def histogram(x, bins=10, range=None, weights=None, density=False):
    """Computes the histogram of a set of data.

    Args:
        x (cupy.ndarray): Input array.
        bins (int or cupy.ndarray): If ``bins`` is an int, it represents the
            number of bins. If ``bins`` is an :class:`~cupy.ndarray`, it
            represents a bin edges.
        range (2-tuple of float, optional): The lower and upper range of the
            bins.  If not provided, range is simply ``(x.min(), x.max())``.
            Values outside the range are ignored. The first element of the
            range must be less than or equal to the second. `range` affects the
            automatic bin computation as well. While bin width is computed to
            be optimal based on the actual data within `range`, the bin count
            will fill the entire range including portions containing no data.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.
        weights (cupy.ndarray, optional): An array of weights, of the same
            shape as `x`.  Each value in `x` only contributes its associated
            weight towards the bin count (instead of 1).
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

    if not isinstance(x, cupy.ndarray):
        raise ValueError("x must be a cupy.ndarray")

    x, weights = _ravel_and_check_weights(x, weights)
    bin_edges = _get_bin_edges(x, bins, range)

    if weights is None:
        y = cupy.zeros(bin_edges.size - 1, dtype='l')
        _histogram_kernel(x, bin_edges, bin_edges.size, y)
    else:
        simple_weights = (
            cupy.can_cast(weights.dtype, cupy.float64) or
            cupy.can_cast(weights.dtype, cupy.complex128)
        )
        if not simple_weights:
            # object dtype such as Decimal are supported in NumPy, but not here
            raise NotImplementedError(
                "only weights with dtype that can be cast to float or complex "
                "are supported")
        if weights.dtype.kind == 'c':
            y = cupy.zeros(bin_edges.size - 1, dtype=cupy.complex128)
            _weighted_histogram_kernel(
                x, bin_edges, bin_edges.size, weights.real, y.real)
            _weighted_histogram_kernel(
                x, bin_edges, bin_edges.size, weights.imag, y.imag)
        else:
            if weights.dtype.kind in 'bui':
                y = cupy.zeros(bin_edges.size - 1, dtype=int)
            else:
                y = cupy.zeros(bin_edges.size - 1, dtype=cupy.float64)
            _weighted_histogram_kernel(
                x, bin_edges, bin_edges.size, weights, y)

    if density:
        db = cupy.array(cupy.diff(bin_edges), cupy.float64)
        return y/db/y.sum(), bin_edges
    return y, bin_edges

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


def digitize(x, bins, right=False):
    """Finds the indices of the bins to which each value in input array belongs.

    .. note::

        In order to avoid device synchronization, digitize does not raise
        an exception when the array is not monotonic

    Args:
        x (cupy.ndarray): Input array.
        bins (cupy.ndarray): Array of bins.
            It has to be 1-dimensional and monotonic increasing or decreasing.
        right (bool):
            Indicates whether the intervals include the right or the left bin
            edge.

    Returns:
        cupy.ndarray: Output array of indices, of same shape as ``x``.

    .. seealso:: :func:`numpy.digitize`
    """
    # This is for NumPy compat, although it works fine
    if x.dtype.kind == 'c':
        raise TypeError('x may not be complex')

    if bins.ndim > 1:
        raise ValueError('object too deep for desired array')
    if bins.ndim < 1:
        raise ValueError('object of too small depth for desired array')

    # As the order of the arguments are reversed, the side must be too.
    side = 'left' if right else 'right'
    return cupy._sorting.search._searchsorted(bins, x, side, None, False)
