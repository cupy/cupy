import cupy
import cupyx

from cupyx.scipy import sparse


def find(A):
    """Returns the indices and values of the nonzero elements of a matrix

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose nonzero
            elements are desired.

    Returns:
        tuple of cupy.ndarray:
            It returns (``I``, ``J``, ``V``). ``I``, ``J``, and ``V`` contain
            respectively the row indices, column indices, and values of the
            nonzero matrix entries.

    .. seealso:: :func:`scipy.sparse.find`
    """
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=True)
    A.sum_duplicates()
    nz_mask = A.data != 0
    return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]


def tril(A, k=0, format=None):
    """Returns the lower triangular portion of a matrix in sparse format

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose lower
            triangular portion is desired.
        k (integer): The top-most diagonal of the lower triangle.
        format (string): Sparse format of the result, e.g. 'csr', 'csc', etc.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Lower triangular portion of A in sparse format.

    .. seealso:: :func:`scipy.sparse.tril`
    """
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=False)
    mask = A.row + k >= A.col
    return _masked_coo(A, mask).asformat(format)


def triu(A, k=0, format=None):
    """Returns the upper triangular portion of a matrix in sparse format

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose upper
            triangular portion is desired.
        k (integer): The bottom-most diagonal of the upper triangle.
        format (string): Sparse format of the result, e.g. 'csr', 'csc', etc.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Upper triangular portion of A in sparse format.

    .. seealso:: :func:`scipy.sparse.triu`
    """
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=False)
    mask = A.row + k <= A.col
    return _masked_coo(A, mask).asformat(format)


def _check_A_type(A):
    if not (isinstance(A, cupy.ndarray) or cupyx.scipy.sparse.isspmatrix(A)):
        msg = 'A must be cupy.ndarray or cupyx.scipy.sparse.spmatrix'
        raise TypeError(msg)


def _masked_coo(A, mask):
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    return sparse.coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)
