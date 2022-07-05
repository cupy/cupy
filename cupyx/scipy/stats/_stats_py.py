import warnings

import cupy


def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']

    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))

    try:
        contains_nan = cupy.isnan(cupy.sum(a))
    except TypeError:
        try:
            contains_nan = cupy.nan in set(a.ravel())
        except TypeError:
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly "
                          "checked for nan values. nan values "
                          "will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy


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


def _quiet_nanmean(x):
    """Compute nanmean for the 1d array x, but quietly return
    nan if x is all nan. The return value is a 1d array with
    length 1, so it can be used in cupy.apply_along_axis.

    """

    y = x[~cupy.isnan(x)]
    if y.size == 0:
        return cupy.array([cupy.nan])
    else:
        return cupy.mean(y, keepdims=True)


def _quiet_nanstd(x, ddof=0):
    """Compute nanstd for the 1d array x, but quietly return
    nan if x is all nan. The return value is a 1d array
    with length 1, so it can be used in cupy.apply_along_axis

    """

    y = x[~cupy.isnan(x)]
    if y.size == 0:
        return cupy.array([cupy.nan])
    else:
        return cupy.std(y, keepdims=True, ddof=ddof)


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

    a = compare

    if a.size == 0:
        return cupy.empty(a.shape)

    contains_nan, nan_policy = _contains_nan(a, nan_policy)

    if contains_nan and nan_policy == 'omit':
        if axis is None:
            mn = _quiet_nanmean(a.ravel())
            std = _quiet_nanstd(a.ravel(), ddof=ddof)
            isconst = _isconst(a.ravel())
        else:
            mn = cupy.apply_along_axis(_quiet_nanmean, axis, a)
            std = cupy.apply_along_axis(_quiet_nanstd, axis, a, ddof=ddof)
            isconst = cupy.apply_along_axis(_isconst, axis, a)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.item(0) == a).all()
        else:
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)

    # Set std deviations that are 0 to 1 to avoid division by 0.
    std[isconst] = 1.0
    z = (scores - mn) / std

    # Set the outputs associated with a constant input to nan.
    z[cupy.broadcast_to(isconst, z.shape)] = cupy.nan
    return z
