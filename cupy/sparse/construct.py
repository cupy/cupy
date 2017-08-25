import cupy


def eye(m, n=None, k=0, dtype='d', format=None):
    """Creates a sparse matrix with ones on diagonal.

    Args:
        m (int): Number of rows.
        n (int or None): Number of columns. If it is ``None``,
            it makes a square matrix.
        k (int): Diagonal to place ones on.
        dtype: Type of a matrix to create.
        format (str or None): Format of the result, e.g. ``format="csr"``.

    Returns:
        cupy.sparse.spmatrix: Created sparse matrix.

    .. seealso:: :func:`scipy.sparse.eye`

    """
    if n is None:
        n = m
    m, n = int(m), int(n)

    if m == n and k == 0:
        if format in ['csr', 'csc']:
            indptr = cupy.arange(n + 1, dtype='i')
            indices = cupy.arange(n, dtype='i')
            data = cupy.ones(n, dtype=dtype)
            if format == 'csr':
                cls = cupy.sparse.csr_matrix
            else:
                cls = cupy.sparse.csc_matrix
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            row = cupy.arange(n, dtype='i')
            col = cupy.arange(n, dtype='i')
            data = cupy.ones(n, dtype=dtype)
            return cupy.sparse.coo_matrix((data, (row, col)), (n, n))

    diags = cupy.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return spdiags(diags, k, m, n).asformat(format)


def identity(n, dtype='d', format=None):
    """Creates an identity matrix in sparse format.

    .. note::
       Currently it only supports csr, csc and coo formats.

    Args:
        n (int): Number of rows and columns.
        dtype: Type of a matrix to create.
        format (str or None): Format of the result, e.g. ``format="csr"``.

    Returns:
        cupy.sparse.spmatrix: Created identity matrix.

    .. seealso:: :func:`scipy.sparse.identity`

    """
    return eye(n, n, dtype=dtype, format=format)


def spdiags(data, diags, m, n, format=None):
    """Creates a sparse matrix from diagonals.

    Args:
        data (cupy.ndarray): Matrix diagonals stored row-wise.
        diags (cupy.ndarray): Diagonals to set.
        m (int): Number of rows.
        n (int): Number of cols.
        format (str or None): Sparse format, e.g. ``format="csr"``.

    Returns:
        cupy.sparse.spmatrix: Created sparse matrix.

    .. seealso:: :func:`scipy.sparse.spdiags`

    """
    return cupy.sparse.dia_matrix((data, diags), shape=(m, n)).asformat(format)
