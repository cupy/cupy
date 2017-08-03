import cupy


def eye(m, n=None, k=0, dtype='d', format=None):
    """Creates a sparse matrix with ones on diagonal.

    .. note::
       Currently it only supports ``m==n``, ``k=0`` and csr, csc and coo
       formats.

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

    raise ValueError(
        'only supports identity matries of csr, csc and coo format')
