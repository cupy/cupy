from __future__ import annotations

import cupy
import cupyx

from cupyx.scipy import sparse


def _is_sparray(A):
    return sparse.issparse(A) and isinstance(A, sparse.sparray)


def find(A):
    """Return the indices and values of nonzero elements of a sparse object.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse): Object whose nonzero elements
            are desired.

    Returns:
        tuple of cupy.ndarray:
            ``(I, J, V)`` containing the row indices, column indices, and
            values of the nonzero entries.

    .. seealso:: :func:`scipy.sparse.find`
    """
    _check_A_type(A)
    use_array = _is_sparray(A)
    coo_cls = sparse.coo_array if use_array else sparse.coo_matrix
    A = coo_cls(A, copy=True)
    A.sum_duplicates()
    nz_mask = A.data != 0
    return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a sparse object.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse): Input array/matrix.
        k (integer): The top-most diagonal of the lower triangle.
        format (string): Sparse format of the result, e.g. ``'csr'``,
            ``'csc'``.

    Returns:
        cupyx.scipy.sparse: Lower triangular portion of ``A``.  Returns a
        sparse array when ``A`` is a sparse array, else a sparse matrix.

    .. seealso:: :func:`scipy.sparse.tril`
    """
    _check_A_type(A)
    use_array = _is_sparray(A)
    coo_cls = sparse.coo_array if use_array else sparse.coo_matrix
    A = coo_cls(A, copy=False)
    mask = A.row + k >= A.col
    return _masked_coo(A, mask, use_array).asformat(format)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a sparse object.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse): Input array/matrix.
        k (integer): The bottom-most diagonal of the upper triangle.
        format (string): Sparse format of the result, e.g. ``'csr'``,
            ``'csc'``.

    Returns:
        cupyx.scipy.sparse: Upper triangular portion of ``A``.  Returns a
        sparse array when ``A`` is a sparse array, else a sparse matrix.

    .. seealso:: :func:`scipy.sparse.triu`
    """
    _check_A_type(A)
    use_array = _is_sparray(A)
    coo_cls = sparse.coo_array if use_array else sparse.coo_matrix
    A = coo_cls(A, copy=False)
    mask = A.row + k <= A.col
    return _masked_coo(A, mask, use_array).asformat(format)


def _check_A_type(A):
    if not (isinstance(A, cupy.ndarray) or cupyx.scipy.sparse.issparse(A)):
        msg = 'A must be cupy.ndarray or cupyx.scipy.sparse'
        raise TypeError(msg)


def _masked_coo(A, mask, use_array=False):
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    coo_cls = sparse.coo_array if use_array else sparse.coo_matrix
    return coo_cls._from_parts(data, row, col, shape=A.shape)
