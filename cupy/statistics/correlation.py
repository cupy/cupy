import warnings

import numpy

import cupy


# TODO(okuta): Implement corrcoef


# TODO(okuta): Implement correlate


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None):
    """Returns a covariance matrix from a given data and weights.

    Args:
        m (cupy.ndarray): 1D or 2D array to compute the covariance matrix.
        y (cupy.ndarray): An additional data to compute.
        rowvar (bool): If ``True``, the function assumes that each row
            represents a variable and each column corresponds to its
            observation. Otherwise, the relationship is transposed.
        bias (bool): If ``False``, the covariance matrix is normalized
            by ``N - 1``, where ``N`` denotes the number of observations.
            Otherwise, the matrix is normalized by ``N``.
        ddof (int): Means Delta Degrees of Freedom. If specified, the divisor
            used in computation is ``N - ddof``. Note that ``bias`` is not
            used when using ``ddof``.
        fweights (cupy.ndarray): 1D integer array containing frequency weights.
        aweights (cupy.ndarray): 1D nonnegative array containing weights of
            the observation vector.

    Returns:
        cupy.ndarray: The covariance matrix of the variables.

    .. seealso:: :func:`numpy.cov`
    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')

    # TODO(msaito): Support complex array with fweights
    if numpy.issubdtype(m.dtype, numpy.complex):
        if fweights is not None:
            raise NotImplementedError(
                'Complex array with fweights is not supported')
        if aweights is not None:
            raise NotImplementedError(
                'Complex array with aweights is not supported')

    # Handles complex arrays too
    m = cupy.asarray(m)
    if m.ndim > 2:
        raise ValueError('m has more than 2 dimensions')

    if y is None:
        dtype = numpy.result_type(m, numpy.float64)
    else:
        y = cupy.asarray(y)
        if y.ndim > 2:
            raise ValueError('y has more than 2 dimensions')
        dtype = numpy.result_type(m, y, numpy.float64)

    X = cupy.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return cupy.array([]).reshape(0, 0)
    if y is not None:
        y = cupy.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = cupy.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        if not issubclass(fweights.dtype.type, numpy.integer):
            raise TypeError('fweights must be integer')
        if fweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional fweights')
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and fweights')
        if any(fweights < 0):
            raise ValueError('fweights cannot be negative')
        w = cupy.asarray(fweights, dtype=float)
    if aweights is not None:
        aweights = cupy.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional aweights')
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and aweights')
        if any(aweights < 0):
            raise ValueError('aweights cannot be negative')
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = cupy.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * cupy.sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice',
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    c = cupy.dot(X, X_T.conj())
    c *= 1. / cupy.float64(fact)
    return c.squeeze()
