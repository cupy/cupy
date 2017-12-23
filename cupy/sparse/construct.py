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


def random(m, n, density=0.01, format='coo', dtype=None,
           random_state=None, data_rvs=None):
    """Generates a random sparse matrix.

    This function generates a random sparse matrix. First it selects non-zero
    elements with given density ``density`` from ``(m, n)`` elements.
    So the number of non-zero elements ``k`` is ``k = m * n * density``.
    Value of each element is selected with ``data_rvs`` function.

    Args:
        m (int): Number of rows.
        n (int): Number of cols.
        density (float): Ratio of non-zero entries.
        format (str): Matrix format.
        dtype (dtype): Type of the returned matrix values.
        random_state (cupy.random.RandomState or int):
            State of random number generator.
            If an integer is given, the method makes a new state for random
            number generator and uses it.
            If it is not given, the default state is used.
            This state is used to generate random indexes for nonzero entries.
        data_rvs (callable): A function to generate data for a random matrix.
            If it is not given, `random_state.rand` is used.

    Returns:
        cupy.sparse.spmatrix: Generated matrix.

    .. seealso:: :func:`scipy.sparse.random`

    """
    if density < 0 or density > 1:
        raise ValueError('density expected to be 0 <= density <= 1')
    dtype = cupy.dtype(dtype)
    if dtype.char not in 'fd':
        raise NotImplementedError('type %s not supported' % dtype)

    mn = m * n

    k = int(density * m * n)

    if random_state is None:
        random_state = cupy.random
    elif isinstance(random_state, (int, cupy.integer)):
        random_state = cupy.random.RandomState(random_state)

    if data_rvs is None:
        data_rvs = random_state.rand

    ind = random_state.choice(mn, size=k, replace=False)
    j = cupy.floor(ind * (1. / m)).astype('i')
    i = ind - j * m
    vals = data_rvs(k).astype(dtype)
    return cupy.sparse.coo_matrix(
        (vals, (i, j)), shape=(m, n)).asformat(format)


def rand(m, n, density=0.01, format='coo', dtype=None, random_state=None):
    """Generates a random sparse matrix.

    See ``cupy.sparse.random`` for detail.

    Args:
        m (int): Number of rows.
        n (int): Number of cols.
        density (float): Ratio of non-zero entries.
        format (str): Matrix format.
        dtype (dtype): Type of the returned matrix values.
        random_state (cupy.random.RandomState or int):
            State of random number generator.
            If an integer is given, the method makes a new state for random
            number generator and uses it.
            If it is not given, the default state is used.
            This state is used to generate random indexes for nonzero entries.

    Returns:
        cupy.sparse.spmatrix: Generated matrix.

    .. seealso:: :func:`scipy.sparse.rand`
    .. seealso:: :func:`cupy.sparse.random`

    """
    return random(m, n, density, format, dtype, random_state)
