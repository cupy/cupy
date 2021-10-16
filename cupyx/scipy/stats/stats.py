# Copyright 2002 Gary Strangman.  All rights reserved
# Copyright 2002-2016 The SciPy Developers
#
# The original code from Gary Strangman was heavily adapted for use in SciPy by
# Travis Oliphant.  The original code came with the following disclaimer:
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties of
# merchantability and fitness for a given application.  In no event shall Gary
# Strangman be liable for any direct, indirect, incidental, special, exemplary
# or consequential damages (including, but not limited to, loss of use, data or
# profits, or business interruption) however caused and on any theory of
# liability, whether in contract, strict liability or tort (including
# negligence or otherwise) arising in any way out of the use of this software,
# even if advised of the possibility of such damage.
"""
A collection of basic statistical functions for Python.

References
----------
.. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.
"""
import cupy as np


def trim_mean(a, proportiontocut, axis=0):
    """Return mean of array after trimming distribution from both tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. The input is sorted before slicing. Slices off less if proportion
    results in a non-integer slice index (i.e., conservatively slices off
    `proportiontocut` ).

    Parameters
    ----------
    a : array_like
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
    >>> import cupy as np
    >>> from cupyx.scipy import stats
    >>> x = np.arange(20)
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
    a = np.asarray(a)

    if a.size == 0:
        return np.nan

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return np.mean(atmp[tuple(sl)], axis=axis)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
