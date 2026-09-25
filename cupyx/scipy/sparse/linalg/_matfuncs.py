from __future__ import annotations

import operator

import cupyx.scipy.sparse


def _pow_by_squaring(A, power, identity):
    """Raise square sparse ``A`` to ``power`` by exponentiation-by-squaring.

    Shared by :func:`matrix_power` and ``spmatrix.__pow__`` so the recursion
    and its validation stay single-sourced.  ``identity`` is a zero-arg
    callable returning the ``power == 0`` result (the two callers differ
    only in that identity: an array vs a matrix).  ``@`` is used for the
    products -- it is matmul for both sparse matrices and arrays.
    """
    M, N = A.shape
    if M != N:
        raise TypeError('sparse matrix is not square')
    try:
        power = operator.index(power)
    except TypeError:
        raise ValueError('exponent must be an integer')
    if power < 0:
        raise ValueError('exponent must be >= 0')
    if power == 0:
        return identity()
    if power == 1:
        return A.copy()
    tmp = _pow_by_squaring(A, power // 2, identity)
    return A @ tmp @ tmp if power % 2 else tmp @ tmp


def matrix_power(A, power):
    """Raise a square matrix to the integer power ``power``.

    For non-negative integers, ``A**power`` is computed using repeated
    matrix multiplications.  Negative integers are not supported.

    Args:
        A (cupyx.scipy.sparse.spmatrix or cupyx.scipy.sparse.sparray):
            Square sparse array or matrix to raise to the power.
        power (int): Non-negative integer exponent.

    Returns:
        cupyx.scipy.sparse.spmatrix or cupyx.scipy.sparse.sparray:
        ``A`` raised to ``power``.  For ``power >= 1`` the result has the
        same shape and class as ``A`` (its sparse format may differ).
        ``power == 0`` returns the identity built by
        :func:`~cupyx.scipy.sparse.eye_array` -- a ``csr_array`` (scipy
        returns a ``dia_array``); like scipy this is an array even when
        ``A`` is a matrix.

    .. seealso:: :func:`scipy.sparse.linalg.matrix_power`
    """
    return _pow_by_squaring(
        A, power,
        lambda: cupyx.scipy.sparse.eye_array(A.shape[0], dtype=A.dtype))
