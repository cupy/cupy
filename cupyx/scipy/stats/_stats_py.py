import cupy


def gmean(a, axis=0, dtype=None, weights=None):
    # This is just the gmean function from scipy,
    # but adapted to use cp functions instead of numpy/scipy ones.
    r"""Compute the weighted geometric mean along the specified axis.

    The weighted geometric mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \exp \left( \frac{ \sum_{i=1}^n w_i \ln a_i }{ \sum_{i=1}^n w_i }
                   \right) \, ,

    and, with equal weights, it gives:

    .. math::

        \sqrt[n]{ \prod_{i=1}^n a_i } \, .

    Parameters
    ----------
    a : cupy.ndarray
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the geometric mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type to which the input arrays are cast before the calculation is
        performed.
    weights : cupy.ndarray, optional
        The `weights` array must be broadcastable to the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    gmean : ndarray
        See `dtype` parameter above.

    Notes
    -----
    The sample geometric mean is the exponential of the mean of the natural
    logarithms of the observations.
    Negative observations will produce NaNs in the output because the *natural*
    logarithm (as opposed to the *complex* logarithm) is defined only for
    non-negative reals.

    References
    ----------
    .. [1] "Weighted Geometric Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Weighted_geometric_mean.
    .. [2] Grossman, J., Grossman, M., Katz, R., "Averages: A New Approach",
           Archimedes Foundation, 1983

    """

    if dtype == cupy.float16:
        # Avoid large numerical errors in float16
        dtype = cupy.float32

    a = cupy.asarray(a, dtype=dtype)
    if weights is not None:
        weights = cupy.asarray(weights, dtype=dtype)

    log_a = cupy.log(a)

    return cupy.exp(cupy.average(log_a, axis=axis, weights=weights))


def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position

    """

    return cupy.take_along_axis(arr, cupy.array(0, ndmin=arr.ndim), axis)


def _isconst(x):
    """Check if all values in x are the same.  nans are ignored.
    x must be a 1d array. The return value is a 1d array
    with length 1, so it can be used in cupy.apply_along_axis.

    """

    y = x[~cupy.isnan(x)]
    if y.size == 0:
        return cupy.array([True])
    else:
        return (y[0] == y).all(keepdims=True)


def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """Compute the z-score.

    Compute the z-score of each value in the sample, relative to
    the sample mean and standard deviation.

    Parameters
    ----------
    a : array-like
        An array like object containing the sample data
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None,
        compute over the whole arrsy `a`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate'
        returns nan, 'raise' throws an error, 'omit' performs
        the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit',
        nans in the input also propagate to the output,
        but they do not affect the z-scores computed
        for the non-nan values

    Returns
    -------
    zscore : array-like
        The z-scores, standardized by mean and standard deviation of
        input array `a`

    """

    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)


def zmap(scores, compare, axis=0, ddof=0, nan_policy='propagate'):
    """Calculate the relative z-scores.

    Return an array of z-scores, i.e., scores that are standardized
    to zero mean and unit variance, where mean and variance are
    calculated from the comparison array.

    Parameters
    ----------
    scores : array-like
        The input for which z-scores are calculated
    compare : array-like
        The input from which the mean and standard deviation of
        the normalization are taken; assumed to have the same
        dimension as `scores`
    axis : int or None, optional
        Axis over which mean and variance of `compare` are calculated.
        Default is 0. If None, compute over the whole array `scores`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle the occurrence of nans in `compare`.
        'propagate' returns nan, 'raise' raises an exception, 'omit'
        performs the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit', nans in `scores`
        also propagate to the output, but they do not affect the z-scores
        computed for the non-nan values

    Returns
    -------
    zscore : array-like
        Z-scores, in the same shape as `scores`

    """

    policies = ['propagate', 'raise', 'omit']

    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))

    a = compare

    if a.size == 0:
        dtype = a.dtype if a.dtype.kind in 'fc' else cupy.float64
        return cupy.empty(a.shape, dtype)

    if nan_policy == 'raise':
        contains_nan = cupy.isnan(cupy.sum(a))

        if contains_nan:  # synchronize!
            raise ValueError("The input contains nan values")

    if nan_policy == 'omit':
        if axis is None:
            mn = cupy.nanmean(a.ravel())
            std = cupy.nanstd(a.ravel(), ddof=ddof)
            isconst = _isconst(a.ravel())
        else:
            mn = cupy.nanmean(a, axis=axis, keepdims=True)
            std = cupy.nanstd(a, axis=axis, keepdims=True, ddof=ddof)
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.ravel()[0] == a).all()
        else:
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)

    # Set std deviations that are 0 to 1 to avoid division by 0.
    std[isconst] = 1.0
    z = (scores - mn) / std

    # Set the outputs associated with a constant input to nan.
    z[cupy.broadcast_to(isconst, z.shape)] = cupy.nan
    return z
