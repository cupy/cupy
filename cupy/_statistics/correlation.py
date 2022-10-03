import functools
import warnings

import numpy

import cupy
from cupy import _core


def corrcoef(a, y=None, rowvar=True, bias=None, ddof=None, *, dtype=None):
    """Returns the Pearson product-moment correlation coefficients of an array.

    Args:
        a (cupy.ndarray): Array to compute the Pearson product-moment
            correlation coefficients.
        y (cupy.ndarray): An additional set of variables and observations.
        rowvar (bool): If ``True``, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is
            transposed.
        bias (None): Has no effect, do not use.
        ddof (None): Has no effect, do not use.
        dtype: Data type specifier. By default, the return data-type will have
            at least `numpy.float64` precision.

    Returns:
        cupy.ndarray: The Pearson product-moment correlation coefficients of
        the input array.

    .. seealso:: :func:`numpy.corrcoef`

    """
    if bias is not None or ddof is not None:
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning)

    out = cov(a, y, rowvar, dtype=dtype)
    try:
        d = cupy.diag(out)
    except ValueError:
        return out / out

    stddev = cupy.sqrt(d.real)
    out /= stddev[:, None]
    out /= stddev[None, :]

    cupy.clip(out.real, -1, 1, out=out.real)
    if cupy.iscomplexobj(out):
        cupy.clip(out.imag, -1, 1, out=out.imag)

    return out


def correlate(a, v, mode='valid'):
    """Returns the cross-correlation of two 1-dimensional sequences.

    Args:
        a (cupy.ndarray): first 1-dimensional input.
        v (cupy.ndarray): second 1-dimensional input.
        mode (str, optional): `valid`, `same`, `full`

    Returns:
        cupy.ndarray: Discrete cross-correlation of a and v.

    .. seealso:: :func:`numpy.correlate`

    """
    if a.size == 0 or v.size == 0:
        raise ValueError('Array arguments cannot be empty')
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError('object too deep for desired array')
    # choose_conv_method does not choose from the values in
    # the input array, so no need to apply conj.
    method = cupy._math.misc._choose_conv_method(a, v, mode)
    if method == 'direct':
        out = cupy._math.misc._dot_convolve(a, v.conj()[::-1], mode)
    elif method == 'fft':
        out = cupy._math.misc._fft_convolve(a, v.conj()[::-1], mode)
    else:
        raise ValueError('Unsupported method')
    return out


def cov(a, y=None, rowvar=True, bias=False, ddof=None,
        fweights=None, aweights=None, *, dtype=None):
    """Returns the covariance matrix of an array.

    This function currently does not support ``fweights`` and ``aweights``
    options.

    Args:
        a (cupy.ndarray): Array to compute covariance matrix.
        y (cupy.ndarray): An additional set of variables and observations.
        rowvar (bool): If ``True``, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is
            transposed.
        bias (bool): If ``False``, normalization is by ``(N - 1)``, where N is
            the number of observations given (unbiased estimate). If ``True``,
            then normalization is by ``N``.
        ddof (int): If not ``None`` the default value implied by bias is
            overridden. Note that ``ddof=1`` will return the unbiased estimate
            and ``ddof=0`` will return the simple average.

        fweights (cupy.ndarray, int): 1-D array of integer frequency weights.
            the number of times each observation vector should be repeated.
            It is required that fweights >= 0. However, the function will not
            error when fweights < 0 for performance reasons.
        aweights (cupy.ndarray): 1-D array of observation vector weights.
            These relative weights are typically large for observations
            considered "important" and smaller for observations considered
            less "important". If ``ddof=0`` the array of weights can be used
            to assign probabilities to observation vectors.
            It is required that aweights >= 0. However, the function will not
            error when aweights < 0 for performance reasons.
        dtype: Data type specifier. By default, the return data-type will have
            at least `numpy.float64` precision.

    Returns:
        cupy.ndarray: The covariance matrix of the input array.

    .. seealso:: :func:`numpy.cov`

    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')

    if a.ndim > 2:
        raise ValueError('Input must be <= 2-d')

    if dtype is None:
        if y is None:
            dtype = numpy.promote_types(a.dtype, numpy.float64)
        else:
            if y.ndim > 2:
                raise ValueError('y must be <= 2-d')
            dtype = functools.reduce(
                numpy.promote_types,
                (a.dtype, y.dtype, numpy.float64)
            )

    X = cupy.array(a, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return cupy.array([]).reshape(0, 0)
    if y is not None:
        y = cupy.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = _core.concatenate_method((X, y), axis=0)

    if ddof is None:
        ddof = 0 if bias else 1

    w = None
    if fweights is not None:
        if not isinstance(fweights, cupy.ndarray):
            raise TypeError(
                "fweights must be a cupy.ndarray")
        if fweights.dtype.char not in 'bBhHiIlLqQ':
            raise TypeError(
                "fweights must be integer")
        fweights = fweights.astype(dtype=float)
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        w = fweights

    if aweights is not None:
        if not isinstance(fweights, cupy.ndarray):
            raise TypeError(
                "aweights must be a cupy.ndarray")
        aweights = aweights.astype(dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
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
        fact = w_sum - ddof * sum(w*aweights) / w_sum

    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice',
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= X.mean(axis=1)[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    out = X.dot(X_T.conj()) * (1 / cupy.float64(fact))

    return out.squeeze()
