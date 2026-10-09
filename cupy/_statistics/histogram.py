from __future__ import annotations

import operator
import warnings

import numpy

import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy._core import _cuda_compute_histogram
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime


# rename builtin range for use in functions that take a range argument
_range = range


# TODO(unno): use searchsorted
_histogram_kernel = _core.ElementwiseKernel(
    'S x, raw T bins, int64 n_bins_',
    'raw U y',
    '''
    using index_t = decltype(bins)::index_t;
    index_t n_bins = static_cast<index_t>(n_bins_);

    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    index_t high = n_bins - 1;
    index_t low = 0;

    while (high - low > 1) {
        index_t mid = low + (high - low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    atomicAdd(&y[low], U(1));
    ''',
    'cupy_histogram_kernel')


_weighted_histogram_kernel = _core.ElementwiseKernel(
    'S x, raw T bins, int64 n_bins_, raw W weights',
    'raw Y y',
    '''
    using index_t = decltype(bins)::index_t;
    index_t n_bins = static_cast<index_t>(n_bins_);

    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    index_t high = n_bins - 1;
    index_t low = 0;

    while (high - low > 1) {
        index_t mid = low + (high - low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    atomicAdd(&y[low], (Y)weights[i]);
    ''',
    'cupy_weighted_histogram_kernel')


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == cupy.bool_:
        warnings.warn('Converting input from {} to {} for compatibility.'
                      .format(a.dtype, cupy.uint8),
                      RuntimeWarning, stacklevel=3)
        a = a.astype(cupy.uint8)

    if weights is not None:
        if not isinstance(weights, cupy.ndarray):
            raise ValueError('weights must be a cupy.ndarray')
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        numpy.byte: numpy.ubyte,
        numpy.short: numpy.ushort,
        numpy.intc: numpy.uintc,
        numpy.int_: numpy.uint,
        numpy.longlong: numpy.ulonglong,
    }
    dt = numpy.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return cupy.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned. The input may be negative python integers so
        # ensure we pass in arrays with the initial dtype (related to NEP 50).
        return cupy.subtract(
            cupy.asarray(a, dtype=dt), cupy.asarray(b, dtype=dt),
            casting='unsafe', dtype=unsigned_dt)


def _ptp(x):
    """
    Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(x, range):
    """Square root histogram bin estimator."""
    return _ptp(x) / numpy.sqrt(x.size)


def _hist_bin_sturges(x, range):
    """Sturges histogram bin estimator."""
    return _ptp(x) / (numpy.log2(x.size) + 1.0)


def _hist_bin_rice(x, range):
    """Rice histogram bin estimator."""
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    """Scott histogram bin estimator."""
    return (24.0 * numpy.pi**0.5 / x.size)**(1.0 / 3.0) * x.std()


def _hist_bin_stone(x, range):
    """
    Histogram bin estimator based on minimizing the estimated integrated
    squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the
    unknown true distribution. The ISE is estimated using cross-validation and
    can be regarded as a generalization of Scott's rule.
    """
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = cupy.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(numpy.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn('The number of bins estimated may be suboptimal.',
                      RuntimeWarning, stacklevel=3)
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for non-normal
    data.
    """
    if x.size > 2:
        sg1 = numpy.sqrt(
            6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = x.std()
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - x.mean()
            cupy.true_divide(temp, sigma, temp)
            cupy.power(temp, 3, temp)
            g1 = temp.mean()
            return _ptp(x) / (
                1.0 + numpy.log2(x.size)
                + numpy.log2(1.0 + abs(g1) / sg1))
    return 0.0


def _hist_bin_fd(x, range):
    """The Freedman-Diaconis histogram bin estimator."""
    iqr = cupy.subtract(*cupy.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x, range):
    """
    Histogram bin estimator that uses the minimum width of a relaxed
    Freedman-Diaconis and Sturges estimators if the FD bin width does not
    result in a large number of bins.
    """
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    sqrt_bw = _hist_bin_sqrt(x, range)
    # heuristic to limit the maximal number of bins
    fd_bw_corrected = max(fd_bw, sqrt_bw / 2)
    return min(fd_bw_corrected, sturges_bw)


# Private dict initialized at module load time
_hist_bin_selectors = {
    'stone': _hist_bin_stone,
    'auto': _hist_bin_auto,
    'doane': _hist_bin_doane,
    'fd': _hist_bin_fd,
    'rice': _hist_bin_rice,
    'scott': _hist_bin_scott,
    'sqrt': _hist_bin_sqrt,
    'sturges': _hist_bin_sturges,
}


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
                'supplied range of [{}, {}] is not finite'.format(
                    first_edge, last_edge))
    elif a.size == 0:
        first_edge = 0
        last_edge = 1
    else:
        first_edge = a.min()
        last_edge = a.max()
        if not (cupy.isfinite(first_edge) and cupy.isfinite(last_edge)):
            raise ValueError(
                'autodetected range of [{}, {}] is not finite'.format(
                    first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range, weights=None):
    """
    Computes the bins used internally by `histogram`.

    Args:
        a (ndarray): Ravelled data array
        bins (int or str or ndarray): Forwarded argument from `histogram`.
        range (None or tuple): Forwarded argument from `histogram`.
        weights (None or ndarray): Ravelled weights array, or None

    Returns:
        bin_edges (ndarray): Array of bin edges
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(
                '{} is not a valid estimator for `bins`'.format(
                    repr(bin_name)))
        if weights is not None:
            raise TypeError(
                'Automated estimation of the number of bins is not supported '
                'for weighted data')

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not keep.all():  # synchronize! when CuPy
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                if numpy.issubdtype(a.dtype, numpy.integer) and width < 1:
                    width = 1
                delta = _unsigned_subtract(last_edge, first_edge)
                n_equal_bins = int(cupy.ceil(delta / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif numpy.ndim(bins) == 1:
        # TODO(okuta): After #3060 is merged, `if cupy.ndim(bins) == 1:`.
        if isinstance(bins, cupy.ndarray):
            bin_edges = bins
        else:
            bin_edges = numpy.asarray(bins)

        if (bin_edges[:-1] > bin_edges[1:]).any():  # synchronize! when CuPy
            raise ValueError(
                '`bins` must increase monotonically, when an array')
        if isinstance(bin_edges, numpy.ndarray):
            bin_edges = cupy.asarray(bin_edges)
        return bin_edges
    elif numpy.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError:
            if isinstance(bins, cupy.ndarray) and bins.dtype.kind in 'bui':
                # CuPy 0-d arrays do not implement ``__index__``
                n_equal_bins = int(bins)  # synchronize! when CuPy
            else:
                raise TypeError(
                    '`bins` must be an integer, a string, or an array')
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges(a, range)
    else:
        raise ValueError('`bins` must be 1d, when an array')

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
        if (bin_edges[:-1] >= bin_edges[1:]).any():  # synchronize!
            raise ValueError(
                'Too many bins for data range. Cannot create {} '
                'finite-sized bins.'.format(n_equal_bins))
    return bin_edges


def histogram_bin_edges(x, bins=10, range=None, weights=None):
    """Calculates only the edges of the bins used by :func:`histogram`.

    Args:
        x (cupy.ndarray): Input array. The histogram is computed over the
            flattened array.
        bins (int or cupy.ndarray or str, optional): If ``bins`` is an int, it
            defines the number of equal-width bins in the given range (10, by
            default). If ``bins`` is a sequence, it defines the bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.

            If ``bins`` is a string from the list below, ``histogram_bin_edges``
            will use the method chosen to calculate the optimal bin width and
            consequently the number of bins from the data that falls within the
            requested range. While the bin width will be optimal for the
            actual data in the range, the number of bins will be computed to
            fill the entire range, including the empty portions. For
            visualisation, using the ``'auto'`` option is suggested. Weighted
            data is not supported for automated bin size selection.

            - ``'auto'``: Minimum bin width between the 'sturges' and 'fd'
              estimators. Provides good all-around performance.
            - ``'fd'`` (Freedman Diaconis Estimator): Robust (resilient to
              outliers) estimator that takes into account data variability and
              data size.
            - ``'doane'``: An improved version of Sturges' estimator that works
              better with non-normal datasets.
            - ``'scott'``: Less robust estimator that takes into account data
              variability and data size.
            - ``'stone'``: Estimator based on leave-one-out cross-validation
              estimate of the integrated squared error. Can be regarded as a
              generalization of Scott's rule.
            - ``'rice'``: Estimator does not take variability into account, only
              data size. Commonly overestimates number of bins required.
            - ``'sturges'``: R's default method, only accounts for data size. Only
              optimal for gaussian data and underestimates number of bins for
              large non-gaussian datasets.
            - ``'sqrt'``: Square root (of data size) estimator, used by Excel and
              other programs for its speed and simplicity.
        range (2-tuple of float, optional): The lower and upper range of the
            bins.  If not provided, range is simply ``(x.min(), x.max())``.
            Values outside the range are ignored. The first element of the
            range must be less than or equal to the second. `range` affects the
            automatic bin computation as well. While bin width is computed to
            be optimal based on the actual data within `range`, the bin count
            will fill the entire range including portions containing no data.
        weights (cupy.ndarray, optional): An array of weights, of the same
            shape as `x`.  Each value in `x` only contributes its associated
            weight towards the bin count (instead of 1). This is currently not
            used by any of the bin estimators, but may be in the future.

    Returns:
        cupy.ndarray: The edges to pass into :func:`histogram`, of dtype float.

    .. note::

        Unlike :func:`numpy.histogram_bin_edges`, ``x`` must already be a
        :class:`cupy.ndarray`; array_like input such as a Python list raises
        ``ValueError``. This matches :func:`cupy.histogram`.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram_bin_edges`
    """  # NOQA: E501
    if not isinstance(x, cupy.ndarray):
        raise ValueError('x must be a cupy.ndarray')

    if x.dtype.kind == 'c':
        # TODO(unno): comparison between complex numbers is not implemented
        raise NotImplementedError('complex number is not supported')

    x, weights = _ravel_and_check_weights(x, weights)
    return _get_bin_edges(x, bins, range, weights)


def histogram(x, bins=10, range=None, density=False, weights=None):
    """Computes the histogram of a set of data.

    Args:
        x (cupy.ndarray): Input array.
        bins (int or str or cupy.ndarray): If ``bins`` is an int, it represents
            the number of bins. If ``bins`` is a string, it represents one of
            the automatic bin width estimators documented in
            :func:`histogram_bin_edges`. If ``bins`` is an
            :class:`~cupy.ndarray`, it represents a bin edges.
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

    .. note::

        Unlike :func:`numpy.histogram`, ``x`` must already be a
        :class:`cupy.ndarray`; array_like input such as a Python list raises
        ``ValueError``.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram`
    """

    if not isinstance(x, cupy.ndarray):
        raise ValueError('x must be a cupy.ndarray')

    if x.dtype.kind == 'c':
        # TODO(unno): comparison between complex numbers is not implemented
        raise NotImplementedError('complex number is not supported')

    x, weights = _ravel_and_check_weights(x, weights)
    bin_edges = _get_bin_edges(x, bins, range, weights)

    if weights is None:
        y = cupy.zeros(bin_edges.size - 1, dtype=cupy.int64)
        for accelerator in _accelerator.get_routine_accelerators():
            # CUB uses int for bin counts
            # TODO(leofang): support >= 2^31 elements in x?
            if (accelerator == _accelerator.ACCELERATOR_CUB
                    and x.size <= 0x7fffffff and bin_edges.size <= 0x7fffffff):
                # Need to ensure the dtype of bin_edges as it's needed for both
                # the CUB call and the correction later
                assert isinstance(bin_edges, cupy.ndarray)
                if numpy.issubdtype(x.dtype, numpy.integer):
                    bin_type = float
                else:
                    bin_type = numpy.result_type(bin_edges.dtype, x.dtype)
                    if (bin_type == numpy.float16 and
                            not common._is_fp16_supported()):
                        bin_type = numpy.float32
                    x = x.astype(bin_type, copy=False)
                acc_bin_edge = bin_edges.astype(bin_type, copy=True)
                # CUB's upper bin boundary is exclusive for all bins, including
                # the last bin, so we must shift it to comply with NumPy
                if x.dtype.kind in 'ui':
                    acc_bin_edge[-1] += 1
                elif x.dtype.kind == 'f':
                    last = acc_bin_edge[-1]
                    acc_bin_edge[-1] = cupy.nextafter(last, last + 1)
                if runtime.is_hip:
                    y = y.astype(cupy.uint64, copy=False)
                out = cub.cub_histogram(x, y, acc_bin_edge)
                if out is None:
                    # fallback to CuPy impl
                    continue
                else:
                    y = out
                if runtime.is_hip:
                    y = y.astype(cupy.int64, copy=False)
                break
        else:
            _histogram_kernel(x, bin_edges, bin_edges.size, y)
    else:
        simple_weights = (
            cupy.can_cast(weights.dtype, cupy.float64) or
            cupy.can_cast(weights.dtype, cupy.complex128)
        )
        if not simple_weights:
            # object dtype such as Decimal are supported in NumPy, but not here
            raise NotImplementedError(
                'only weights with dtype that can be cast to float or complex '
                'are supported')
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
        return y / db / y.sum(), bin_edges
    return y, bin_edges


def histogramdd(sample, bins=10, range=None, weights=None, density=False):
    """Compute the multidimensional histogram of some data.

    Args:
        sample (cupy.ndarray): The data to be histogrammed. (N, D) or (D, N)
            array

            Note the unusual interpretation of sample when an array_like:

            * When an array, each row is a coordinate in a D-dimensional
              space - such as ``histogramdd(cupy.array([p1, p2, p3]))``.
            * When an array_like, each element is the list of values for single
              coordinate - such as ``histogramdd((X, Y, Z))``.

            The first form should be preferred.
        bins (int or tuple of int or cupy.ndarray): The bin specification:

            * A sequence of arrays describing the monotonically increasing bin
              edges along each dimension.
            * The number of bins for each dimension (nx, ny, ... =bins)
            * The number of bins for all dimensions (nx=ny=...=bins).
        range (sequence, optional): A sequence of length D, each an optional
            (lower, upper) tuple giving the outer bin edges to be used if the
            edges are not given explicitly in `bins`. An entry of None in the
            sequence results in the minimum and maximum values being used for
            the corresponding dimension. The default, None, is equivalent to
            passing a tuple of D None values.
        weights (cupy.ndarray): An array of values `w_i` weighing each sample
            `(x_i, y_i, z_i, ...)`. The values of the returned histogram are
            equal to the sum of the weights belonging to the samples falling
            into each bin.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.

    Returns:
        tuple:
        H (cupy.ndarray):
            The multidimensional histogram of sample x. See
            normed and weights for the different possible semantics.
        edges (list of cupy.ndarray):
            A list of D arrays describing the bin
            edges for each dimension.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogramdd`
    """
    if isinstance(sample, cupy.ndarray):
        # Sample is an ND-array.
        if sample.ndim == 1:
            sample = sample[:, cupy.newaxis]
        nsamples, ndim = sample.shape
    else:
        sample = cupy.stack(sample, axis=-1)
        nsamples, ndim = sample.shape

    nbin = numpy.empty(ndim, int)
    edges = ndim * [None]
    dedges = ndim * [None]
    if weights is not None:
        weights = cupy.asarray(weights)

    try:
        nbins = len(bins)
        if nbins != ndim:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.'
            )
    except TypeError:
        # bins is an integer
        bins = ndim * [bins]

    # normalize the range argument
    if range is None:
        range = (None,) * ndim
    elif len(range) != ndim:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(ndim):
        if cupy.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i)
                )
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            num = int(bins[i] + 1)  # synchronize!
            edges[i] = cupy.linspace(smin, smax, num)
        elif cupy.ndim(bins[i]) == 1:
            if not isinstance(bins[i], cupy.ndarray):
                raise ValueError('array-like bins not supported')
            edges[i] = bins[i]
            if (edges[i][:-1] > edges[i][1:]).any():  # synchronize!
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an '
                    'array'.format(i)
                )
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i)
            )

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = cupy.diff(edges[i])

    # Compute the bin number each sample falls into.
    ncount = tuple(
        # avoid cupy.digitize to work around NumPy issue gh-11022
        cupy.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(ndim)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(ndim):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = cupy.ravel_multi_index(ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = cupy.bincount(xy, weights, minlength=numpy.prod(nbin))

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in NumPy gh-7845, for now.
    hist = hist.astype(float)  # Note: NumPy uses casting='safe' here too

    # Remove outliers (indices 0 and -1 for each dimension).
    core = ndim * (slice(1, -1),)
    hist = hist[core]

    if density:
        # calculate the probability density function
        s = hist.sum()
        for i in _range(ndim):
            shape = [1] * ndim
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if any(hist.shape != numpy.asarray(nbin) - 2):
        raise RuntimeError('Internal Shape Error')
    return hist, edges


def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
    """Compute the bi-dimensional histogram of two data samples.

    Args:
        x (cupy.ndarray): The first array of samples to be histogrammed.
        y (cupy.ndarray): The second array of samples to be histogrammed.
        bins (int or tuple of int or cupy.ndarray): The bin specification:

            * A sequence of arrays describing the monotonically increasing bin
              edges along each dimension.
            * The number of bins for each dimension (nx, ny)
            * The number of bins for all dimensions (nx=ny=bins).
        range (sequence, optional): A sequence of length two, each an optional
            (lower, upper) tuple giving the outer bin edges to be used if the
            edges are not given explicitly in `bins`. An entry of None in the
            sequence results in the minimum and maximum values being used for
            the corresponding dimension. The default, None, is equivalent to
            passing a tuple of two None values.
        weights (cupy.ndarray): An array of values `w_i` weighing each sample
            `(x_i, y_i)`. The values of the returned histogram are equal to the
            sum of the weights belonging to the samples falling into each bin.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.

    Returns:
        tuple:
        H (cupy.ndarray):
            The multidimensional histogram of sample x. See
            normed and weights for the different possible semantics.
        edges0 (tuple of cupy.ndarray):
            A list of D arrays describing the bin
            edges for the first dimension.
        edges1 (tuple of cupy.ndarray):
            A list of D arrays describing the bin
            edges for the second dimension.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram2d`
    """
    try:
        n = len(bins)
    except TypeError:
        n = 1

    if n != 1 and n != 2:
        if isinstance(bins, cupy.ndarray):
            xedges = yedges = bins
            bins = [xedges, yedges]
        else:
            raise ValueError('array-like bins not supported in CuPy')

    hist, edges = histogramdd([x, y], bins, range, weights, density)
    return hist, edges[0], edges[1]


_bincount_kernel = _core.ElementwiseKernel(
    'S x', 'raw U bin',
    'atomicAdd(&bin[x], U(1))',
    'cupy_bincount_kernel')


_bincount_with_weight_kernel = _core.ElementwiseKernel(
    'S x, T w', 'raw U bin',
    'atomicAdd(&bin[x], w)',
    'cupy_bincount_with_weight_kernel')


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

    if x.size == 0:
        size = minlength if minlength is not None else 0
        # NumPy returns intp dtype for empty input even when weights is given
        return cupy.zeros((size,), dtype=numpy.intp)

    size = int(cupy.max(x)) + 1  # synchronize!
    if minlength is not None:
        size = max(size, minlength)

    if weights is None:
        b = cupy.zeros((size,), dtype=numpy.intp)

        accelerators = ([] if runtime.is_hip
                        else _accelerator.get_routine_accelerators())
        for accelerator in accelerators:
            if accelerator == _accelerator.ACCELERATOR_CUDA_COMPUTE:
                out = _cuda_compute_histogram.cuda_compute_bincount(
                    x, b, size)
                if out is None:
                    continue
                else:
                    b = out
                    break
            # CUB uses int for bin counts
            # TODO(leofang): support >= 2^31 elements in x?
            if (accelerator == _accelerator.ACCELERATOR_CUB
                    and x.size <= 0x7fffffff and size <= 0x7fffffff):
                out = cub.cub_histogram(x, b, size+1)
                if out is None:
                    continue
                else:
                    b = out
                    break
        else:
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
    """  # NOQA
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
