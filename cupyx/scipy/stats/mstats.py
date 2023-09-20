
import cupy


def mquantiles(a, prob=list([.25, .5, .75]), alphap=.4, betap=.4, axis=None,
               limit=()):
    """
    Computes empirical quantiles for a data array.

    Samples quantile are defined by ``Q(p) = (1-gamma)*x[j] + gamma*x[j+1]``,
    where ``x[j]`` is the j-th order statistic, and gamma is a function of
    ``j = floor(n*p + m)``, ``m = alphap + p*(1 - alphap - betap)`` and
    ``g = n*p + m - j``.

    Reinterpreting the above equations to compare to **R** [1]_, [2]_ lead
    to the equation: ``p(k) = (k - alphap)/(n + 1 - alphap - betap)``

    Typical values of (alphap,betap) are:
        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
          (**R** type 4)
        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
          (**R** type 5)
        - (0,0)    : ``p(k) = k/(n+1)`` :
          (**R** type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
          (**R** type 7, **R** default)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x.
          (**R** type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed
          (**R** type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    Parameters
    ----------
    a : array_like
        Input data, as a sequence or array of dimension at most 2.
    prob : array_like, optional
        List of quantiles to compute.
    alphap : float, optional
        Plotting positions parameter, default is 0.4.
    betap : float, optional
        Plotting positions parameter, default is 0.4.
    axis : int, optional
        Axis along which to perform the trimming.
        If None (default), the input array is first flattened.
    limit : tuple, optional
        Tuple of (lower, upper) values.
        Values of `a` outside this open interval are ignored.

    Returns
    -------
    mquantiles : ndarray
        An array containing the calculated quantiles.

    Notes
    -----
    This formulation is very similar to **R** except the calculation of
    ``m`` from ``alphap`` and ``betap``, where in **R** ``m`` is defined
    with each type.

    This function differs from SciPy since it does not make use of masked
    arrays (which are not available in CuPy), instead masked values will
    be marked with NaNs.

    References
    ----------
    .. [1] *R* statistical software: https://www.r-project.org/
    .. [2] *R* ``quantile`` function:
            http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy.stats.mstats import mquantiles
    >>> a = cp.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    >>> mquantiles(a)
    array([ 19.2,  40. ,  42.8])

    Using a 2D array, specifying axis and limit.

    >>> data = cp.array([[   6.,    7.,    1.],
    ...                  [  47.,   15.,    2.],
    ...                  [  49.,   36.,    3.],
    ...                  [  15.,   39.,    4.],
    ...                  [  42.,   40., -999.],
    ...                  [  41.,   41., -999.],
    ...                  [   7., -999., -999.],
    ...                  [  39., -999., -999.],
    ...                  [  43., -999., -999.],
    ...                  [  40., -999., -999.],
    ...                  [  36., -999., -999.]])
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.2  14.6   1.45]
     [40.   37.5   2.5 ]
     [42.8  40.05  3.55]]

    >>> data[:, 2] = -999.
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.2  14.6    nan]
     [40.   37.5    nan]
     [42.8  40.05   nan]]

    """
    def _quantiles1D(data, mask, m, p, axis=-1):
        x = cupy.sort(data, axis=axis)
        n = cupy.sum(mask, axis=axis).reshape(-1, 1)
        aleph = (n * p + m)
        k = cupy.floor(aleph.clip(1, n - 1)).astype(int)
        gamma = (aleph - k).clip(0, 1)
        indexed_x = cupy.take_along_axis(x, k - 1, axis=axis)
        indexed_x1 = cupy.take_along_axis(x, k, axis=axis)
        return (1. - gamma) * indexed_x + gamma * indexed_x1

    data = a.copy()

    if data.ndim > 2:
        raise TypeError("Array should be 2D at most !")

    mask = cupy.full_like(a, True, dtype=cupy.bool_)

    if limit:
        condition = (data <= limit[0]) | (data >= limit[1])
        data[condition] = cupy.nan
        mask[condition] = False

    p = cupy.array(prob, copy=False, ndmin=1)
    m = alphap + p * (1. - alphap - betap)

    # Computes quantiles along axis (or globally)
    if (axis is None):
        data = data.reshape(1, -1)
        mask = mask.reshape(1, -1)
        axis = -1

    transpose = False
    if axis == 0:
        transpose = True
        data = cupy.atleast_2d(data.T)
        mask = cupy.atleast_2d(mask.T)
        axis = -1

    out = _quantiles1D(data, mask, m, p, axis)
    if transpose:
        out = out.T

    if not out.flags.c_contiguous:
        out = out.copy()
    return out


def scoreatpercentile(data, per, limit=(), alphap=.4, betap=.4):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantiles

    """
    if (per < 0) or (per > 100.):
        raise ValueError("The percentile should be between 0. and 100. !"
                         " (got %s)" % per)

    return mquantiles(data, prob=[per/100.], alphap=alphap, betap=betap,
                      limit=limit, axis=0).squeeze()
