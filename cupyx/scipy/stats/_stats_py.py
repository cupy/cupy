import cupy


def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position"""

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


def zscore(a, axis=0, ddof=0, nan_policy="propagate"):
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


def zmap(scores, compare, axis=0, ddof=0, nan_policy="propagate"):
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

    policies = ["propagate", "raise", "omit"]

    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join(
                "'%s'" % s for s in policies)
        )

    a = compare

    if a.size == 0:
        dtype = a.dtype if a.dtype.kind in "fc" else cupy.float64
        return cupy.empty(a.shape, dtype)

    if nan_policy == "raise":
        contains_nan = cupy.isnan(cupy.sum(a))

        if contains_nan:  # synchronize!
            raise ValueError("The input contains nan values")

    if nan_policy == "omit":
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


def rankdata(a, method="average", *, axis=None, nan_policy="propagate"):
    """Assign ranks to data, dealing with ties appropriately.

    By default (``axis=None``), the data array is first flattened, and a flat
    array of ranks is returned. Separately reshape the rank array to the
    shape of the data array if desired (see Examples).

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):

          * 'average': The average of the ranks that would have been assigned
             to all the tied values is assigned to each value.
          * 'min': The minimum of the ranks that would have been assigned to
             all the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
          * 'max': The maximum of the ranks that would have been assigned to
             all the tied values is assigned to each value.
          * 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
          * 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
    axis : {None, int}, optional
        Axis along which to perform the ranking. If ``None``, the data array
        is first flattened.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': propagates nans through the rank calculation
          * 'omit': performs the calculations ignoring nan values
          * 'raise': raises an error

        .. note::

            When `nan_policy` is 'propagate', the output is an array of *all*
            nans because ranks relative to nans in the input are undefined.
            When `nan_policy` is 'omit', nans in `a` are ignored when ranking
            the other values, and the corresponding locations of the output
            are nan.

        .. versionadded:: 1.10

    Returns
    -------
    ranks : ndarray
         An array of size equal to the size of `a`, containing rank
         scores.

    References
    ----------
    .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking


    """

    methods = ("average", "min", "max", "dense", "ordinal")
    if method not in methods:
        raise ValueError(f'unknown method "{method}"')

    x = cupy.asarray(a)

    if axis is None:
        x = x.ravel()
        axis = -1

    if x.size == 0:
        dtype = float if method == "average" else cupy.dtype("long")
        return cupy.empty(x.shape, dtype=dtype)

    policies = ["propagate", "raise", "omit"]
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join(
                "'%s'" % s for s in policies)
        )

    contains_nan = cupy.isnan(cupy.sum(x))

    x = cupy.swapaxes(x, axis, -1)
    ranks = _rankdata(x, method)

    if contains_nan:
        i_nan = cupy.isnan(
            x) if nan_policy == "omit" else cupy.isnan(x).any(axis=-1)
        ranks = ranks.astype(float, copy=False)
        ranks[i_nan] = cupy.nan

    ranks = cupy.swapaxes(ranks, axis, -1)
    return ranks


def _order_ranks(ranks, j):
    # Reorder ascending order `ranks` according to `j`
    ordered_ranks = cupy.empty(j.shape, dtype=ranks.dtype)
    cupy.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    return ordered_ranks


def _rankdata(x, method, return_ties=False):
    # Rank data `x` by desired `method`; `return_ties` if desired
    shape = x.shape

    # Get sort order
    j = cupy.argsort(x, axis=-1, kind="stable")
    ordinal_ranks = cupy.broadcast_to(
        cupy.arange(1, shape[-1] + 1, dtype=int), shape)

    # Ordinal ranks is very easy because ties don't matter. We're done.
    if method == "ordinal":
        return _order_ranks(ordinal_ranks, j)  # never return ties

    # Sort array
    y = cupy.take_along_axis(x, j, axis=-1)
    # Logical indices of unique elements
    i = cupy.concatenate(
        [cupy.ones(shape[:-1] + (1,), dtype=cupy.bool_),
         y[..., :-1] != y[..., 1:]],
        axis=-1,
    )

    # Integer indices of unique elements
    indices = cupy.arange(y.size)[i.ravel()]
    # Counts of unique elements
    counts = cupy.diff(indices, append=y.size)

    # Compute `'min'`, `'max'`, and `'mid'` ranks of unique elements
    if method == "min":
        ranks = ordinal_ranks[i]
    elif method == "max":
        ranks = ordinal_ranks[i] + counts - 1
    elif method == "average":
        ranks = ordinal_ranks[i] + (counts - 1) / 2
    elif method == "dense":
        ranks = cupy.cumsum(i, axis=-1)[i]

    ranks = cupy.repeat(ranks, counts.tolist()).reshape(shape)
    ranks = _order_ranks(ranks, j)

    if return_ties:
        t = cupy.zeros(shape, dtype=float)
        t[i] = counts
        return ranks, t
    return ranks
