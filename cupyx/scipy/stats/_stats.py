"""
A collection of basic statistical functions for Python.

References
----------
.. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.
"""
import cupy as cp
from cupy._core import _routines_statistics as _statistics


def trim_mean(a, proportiontocut, axis=0):
    """Return mean of array after trimming distribution from both tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. The input is sorted before slicing. Slices off less if proportion
    results in a non-integer slice index (i.e., conservatively slices off
    `proportiontocut` ).

    Parameters
    ----------
    a : cupy.ndarray
        Input array.
    proportiontocut : float
        Fraction to cut off of both tails of the distribution.
    axis : int or None, optional
        Axis along which the trimmed means are computed. Default is 0.
        If None, compute over the whole array `a`.

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    See Also
    --------
    trimboth
    tmean : Compute the trimmed mean ignoring values outside given `limits`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy import stats
    >>> x = cp.arange(20)
    >>> stats.trim_mean(x, 0.1)
    array(9.5)
    >>> x2 = x.reshape(5, 4)
    >>> x2
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> stats.trim_mean(x2, 0.25)
    array([ 8.,  9., 10., 11.])
    >>> stats.trim_mean(x2, 0.25, axis=1)
    array([ 1.5,  5.5,  9.5, 13.5, 17.5])
    """
    if a.size == 0:
        return cp.nan

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = cp.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return cp.mean(atmp[tuple(sl)], axis=axis)


def tmin(a, lowerlimit=None, axis=0, inclusive=True, nan_policy='propagate'):
    """Compute the trimmed minimum.

    This function finds the miminum value of an array `a` along the
    specified axis, but only considering values greater than a specified
    lower limit.

    Parameters
    ----------
    a : cupy.ndarray
        Array of values.
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the lower limit
        are included.  The default value is True.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    tmin : float, int or ndarray
        Trimmed minimum.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy import stats
    >>> x = cp.arange(20)
    >>> stats.tmin(x)
    0

    >>> stats.tmin(x, 13)
    13

    >>> stats.tmin(x, 13, inclusive=False)
    14

    """
    if a.size == 0:
        raise ValueError("No array values within given limits")
    if axis is None:
        a = a.ravel()
        axis = 0
    if a.ndim == 0:
        a = cp.atleast_1d(a).astype(a.dtype)
   
    policies = ['propagate', 'raise', 'omit']

    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))

    lowerlimit=cp.atleast_1d( -cp.inf if lowerlimit is None else lowerlimit)
    lowerlimit = lowerlimit.astype(cp.float16) if a.dtype==cp.float16 else lowerlimit

    inf=cp.iinfo(a.dtype).max if cp.dtype(a.dtype).kind in 'iu' else cp.inf
    contains_nan = cp.isnan(cp.sum(a))
    
    if contains_nan :
            if nan_policy != 'propagate':
                if nan_policy == 'raise':
                    raise ValueError("The input contains nan values")
                return _statistics._tmin(a.astype(cp.float64),lowerlimit,inclusive, False, inf,axis=axis).astype(a.dtype, copy=False)
            return _statistics._tmin(a.astype(cp.float64), lowerlimit, inclusive, True, inf,axis=axis).astype(a.dtype, copy=False)
    else:
       
        if lowerlimit==-cp.inf:
            return cp.amin(a, axis=axis)
        max_element= cp.max(a,keepdims=True)
        if max_element>lowerlimit or (cp.allclose(max_element,lowerlimit) and inclusive):
            return _statistics._tmin(a.astype(cp.float64), lowerlimit, inclusive, False,inf, axis=axis).astype(a.dtype, copy=False)
        raise ValueError("No array values within given limits")