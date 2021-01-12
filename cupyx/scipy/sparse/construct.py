import numpy
import cupy
from cupyx.scipy.sparse import coo
from cupyx.scipy.sparse import csc
from cupyx.scipy.sparse import csr
from cupyx.scipy.sparse import dia
from cupyx.scipy.sparse import sputils


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
        cupyx.scipy.sparse.spmatrix: Created sparse matrix.

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
                cls = csr.csr_matrix
            else:
                cls = csc.csc_matrix
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            row = cupy.arange(n, dtype='i')
            col = cupy.arange(n, dtype='i')
            data = cupy.ones(n, dtype=dtype)
            return coo.coo_matrix((data, (row, col)), (n, n))

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
        cupyx.scipy.sparse.spmatrix: Created identity matrix.

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
        cupyx.scipy.sparse.spmatrix: Created sparse matrix.

    .. seealso:: :func:`scipy.sparse.spdiags`

    """
    return dia.dia_matrix((data, diags), shape=(m, n)).asformat(format)


def _compressed_sparse_stack(blocks, axis):
    """Fast path for stacking CSR/CSC matrices
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = cupy.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    idx_dtype = sputils.get_index_dtype(arrays=[b.indptr for b in blocks],
                                        maxval=max(data.size, constant_dim))
    indices = cupy.empty(data.size, dtype=idx_dtype)
    indptr = cupy.empty(sum(b.shape[axis]
                            for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError(
                'incompatible dimensions for axis %d' % other_axis)
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    if axis == 0:
        return csr.csr_matrix((data, indices, indptr),
                              shape=(sum_dim, constant_dim))
    else:
        return csc.csc_matrix((data, indices, indptr),
                              shape=(constant_dim, sum_dim))


def hstack(blocks, format=None, dtype=None):
    """Stacks sparse matrices horizontally (column wise)

    Args:
        blocks (sequence of cupyx.scipy.sparse.spmatrix):
            sparse matrices to stack

        format (str):
            sparse format of the result (e.g. "csr")
            by default an appropriate sparse matrix format is returned.
            This choice is subject to change.
        dtype (dtype, optional):
            The data-type of the output matrix.  If not given, the dtype is
            determined from that of ``blocks``.

    Returns:
        cupyx.scipy.sparse.spmatrix: the stacked sparse matrix

    .. seealso:: :func:`scipy.sparse.hstack`

    Examples:
        >>> from cupy import array
        >>> from cupyx.scipy.sparse import csr_matrix, hstack
        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))
        >>> B = csr_matrix(array([[5.], [6.]]))
        >>> hstack([A, B]).toarray()
        array([[1., 2., 5.],
               [3., 4., 6.]])
    """
    return bmat([blocks], format=format, dtype=dtype)


def vstack(blocks, format=None, dtype=None):
    """Stacks sparse matrices vertically (row wise)

    Args:
        blocks (sequence of cupyx.scipy.sparse.spmatrix)
            sparse matrices to stack
        format (str, optional):
            sparse format of the result (e.g. "csr")
            by default an appropriate sparse matrix format is returned.
            This choice is subject to change.
        dtype (dtype, optional):
            The data-type of the output matrix.  If not given, the dtype is
            determined from that of `blocks`.

    Returns:
        cupyx.scipy.sparse.spmatrix: the stacked sparse matrix

    .. seealso:: :func:`scipy.sparse.vstack`

    Examples:
        >>> from cupy import array
        >>> from cupyx.scipy.sparse import csr_matrix, vstack
        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))
        >>> B = csr_matrix(array([[5., 6.]]))
        >>> vstack([A, B]).toarray()
        array([[1., 2.],
               [3., 4.],
               [5., 6.]])
    """
    return bmat([[b] for b in blocks], format=format, dtype=dtype)


def bmat(blocks, format=None, dtype=None):
    """Builds a sparse matrix from sparse sub-blocks

    Args:
        blocks (array_like):
            Grid of sparse matrices with compatible shapes.
            An entry of None implies an all-zero matrix.
        format ({'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional):
            The sparse format of the result (e.g. "csr").  By default an
            appropriate sparse matrix format is returned.
            This choice is subject to change.
        dtype (dtype, optional):
            The data-type of the output matrix.  If not given, the dtype is
            determined from that of `blocks`.
    Returns:
        bmat (sparse matrix)

    .. seealso:: :func:`scipy.sparse.bmat`

    Examples:
        >>> from cupy import array
        >>> from cupyx.scipy.sparse import csr_matrix, bmat
        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))
        >>> B = csr_matrix(array([[5.], [6.]]))
        >>> C = csr_matrix(array([[7.]]))
        >>> bmat([[A, B], [None, C]]).toarray()
        array([[1., 2., 5.],
               [3., 4., 6.],
               [0., 0., 7.]])
        >>> bmat([[A, None], [None, C]]).toarray()
        array([[1., 2., 0.],
               [3., 4., 0.],
               [0., 0., 7.]])

    """

    # We assume here that blocks will be 2-D so we need to look, at most,
    # 2 layers deep for the shape
    # TODO(Corey J. Nolet): Check this assumption and raise ValueError

    # NOTE: We can't follow scipy exactly here
    # since we don't have an `object` datatype
    M = len(blocks)
    N = len(blocks[0])

    blocks_flat = []
    for m in range(M):
        for n in range(N):
            if blocks[m][n] is not None:
                blocks_flat.append(blocks[m][n])

    if len(blocks_flat) == 0:
        return coo.coo_matrix((0, 0), dtype=dtype)

    # check for fast path cases
    if (N == 1 and format in (None, 'csr') and
            all(isinstance(b, csr.csr_matrix)
                for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 0)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif (M == 1 and format in (None, 'csc')
          and all(isinstance(b, csc.csc_matrix) for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 1)
        if dtype is not None:
            A = A.astype(dtype)
        return A

    block_mask = numpy.zeros((M, N), dtype=bool)
    brow_lengths = numpy.zeros(M+1, dtype=numpy.int64)
    bcol_lengths = numpy.zeros(N+1, dtype=numpy.int64)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                A = coo.coo_matrix(blocks[i][j])
                blocks[i][j] = A
                block_mask[i][j] = True

                if brow_lengths[i+1] == 0:
                    brow_lengths[i+1] = A.shape[0]
                elif brow_lengths[i+1] != A.shape[0]:
                    msg = ('blocks[{i},:] has incompatible row dimensions. '
                           'Got blocks[{i},{j}].shape[0] == {got}, '
                           'expected {exp}.'.format(i=i, j=j,
                                                    exp=brow_lengths[i+1],
                                                    got=A.shape[0]))
                    raise ValueError(msg)

                if bcol_lengths[j+1] == 0:
                    bcol_lengths[j+1] = A.shape[1]
                elif bcol_lengths[j+1] != A.shape[1]:
                    msg = ('blocks[:,{j}] has incompatible row dimensions. '
                           'Got blocks[{i},{j}].shape[1] == {got}, '
                           'expected {exp}.'.format(i=i, j=j,
                                                    exp=bcol_lengths[j+1],
                                                    got=A.shape[1]))
                    raise ValueError(msg)

    nnz = sum(block.nnz for block in blocks_flat)
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks_flat]
        dtype = sputils.upcast(*all_dtypes) if all_dtypes else None

    row_offsets = numpy.cumsum(brow_lengths)
    col_offsets = numpy.cumsum(bcol_lengths)

    shape = (row_offsets[-1], col_offsets[-1])

    data = cupy.empty(nnz, dtype=dtype)
    idx_dtype = sputils.get_index_dtype(maxval=max(shape))
    row = cupy.empty(nnz, dtype=idx_dtype)
    col = cupy.empty(nnz, dtype=idx_dtype)

    nnz = 0
    ii, jj = numpy.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[int(i)][int(j)]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = B.row + row_offsets[i]
        col[idx] = B.col + col_offsets[j]
        nnz += B.nnz

    return coo.coo_matrix((data, (row, col)), shape=shape).asformat(format)


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
        dtype (~cupy.dtype): Type of the returned matrix values.
        random_state (cupy.random.RandomState or int):
            State of random number generator.
            If an integer is given, the method makes a new state for random
            number generator and uses it.
            If it is not given, the default state is used.
            This state is used to generate random indexes for nonzero entries.
        data_rvs (callable): A function to generate data for a random matrix.
            If it is not given, `random_state.rand` is used.

    Returns:
        cupyx.scipy.sparse.spmatrix: Generated matrix.

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
    return coo.coo_matrix(
        (vals, (i, j)), shape=(m, n)).asformat(format)


def rand(m, n, density=0.01, format='coo', dtype=None, random_state=None):
    """Generates a random sparse matrix.

    See :func:`cupyx.scipy.sparse.random` for detail.

    Args:
        m (int): Number of rows.
        n (int): Number of cols.
        density (float): Ratio of non-zero entries.
        format (str): Matrix format.
        dtype (~cupy.dtype): Type of the returned matrix values.
        random_state (cupy.random.RandomState or int):
            State of random number generator.
            If an integer is given, the method makes a new state for random
            number generator and uses it.
            If it is not given, the default state is used.
            This state is used to generate random indexes for nonzero entries.

    Returns:
        cupyx.scipy.sparse.spmatrix: Generated matrix.

    .. seealso:: :func:`scipy.sparse.rand`
    .. seealso:: :func:`cupyx.scipy.sparse.random`

    """
    return random(m, n, density, format, dtype, random_state)


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """Construct a sparse matrix from diagonals.

    Args:
        diagonals (sequence of array_like):
            Sequence of arrays containing the matrix diagonals, corresponding
            to `offsets`.
        offsets (sequence of int or an int):
            Diagonals to set:
                - k = 0  the main diagonal (default)
                - k > 0  the k-th upper diagonal
                - k < 0  the k-th lower diagonal
        shape (tuple of int):
            Shape of the result. If omitted, a square matrix large enough
            to contain the diagonals is returned.
        format ({"dia", "csr", "csc", "lil", ...}):
            Matrix format of the result.  By default (format=None) an
            appropriate sparse matrix format is returned.  This choice is
            subject to change.
        dtype (dtype): Data type of the matrix.

    Returns:
        cupyx.scipy.sparse.spmatrix: Generated matrix.

    Notes:
        This function differs from `spdiags` in the way it handles
        off-diagonals.

        The result from `diags` is the sparse equivalent of::

            cupy.diag(diagonals[0], offsets[0])
            + ...
            + cupy.diag(diagonals[k], offsets[k])

        Repeated diagonal offsets are disallowed.
    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if sputils.isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or sputils.isscalarlike(diagonals[0]):
            diagonals = [cupy.atleast_1d(diagonals)]
        else:
            raise ValueError('Different number of diagonals and offsets.')
    else:
        diagonals = list(map(cupy.atleast_1d, diagonals))

    if isinstance(offsets, cupy.ndarray):
        offsets = offsets.get()
    offsets = numpy.atleast_1d(offsets)

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError('Different number of diagonals and offsets.')

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        dtype = cupy.common_type(*diagonals)

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = cupy.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError(
                'Offset %d (index %d) out of bounds' % (offset, j))
        try:
            data_arr[j, k:k+length] = diagonal[..., :length]
        except ValueError:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    'Diagonal length (index %d: %d at offset %d) does not '
                    'agree with matrix size (%d, %d).' % (
                        j, len(diagonal), offset, m, n))
            raise

    return dia.dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)


def kron(A, B, format=None):
    """Kronecker product of sparse matrices A and B.

    Args:
        A (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        B (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        format (str): the format of the returned sparse matrix.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Generated sparse matrix with the specified ``format``.

    .. seealso:: :func:`scipy.sparse.kron`

    """
    # TODO(leofang): support BSR format when it's added to CuPy
    # TODO(leofang): investigate if possible to optimize performance by
    #                starting with CSR instead of COO matrices

    A = coo.coo_matrix(A)
    B = coo.coo_matrix(B)
    out_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # kronecker product is the zero matrix
        return coo.coo_matrix(out_shape).asformat(format)

    if max(out_shape[0], out_shape[1]) > cupy.iinfo('int32').max:
        dtype = cupy.int64
    else:
        dtype = cupy.int32

    # expand entries of A into blocks
    row = A.row.astype(dtype, copy=True) * B.shape[0]
    row = row.repeat(B.nnz)
    col = A.col.astype(dtype, copy=True) * B.shape[1]
    col = col.repeat(B.nnz)
    data = A.data.repeat(B.nnz)  # data's dtype follows that of A in SciPy

    # increment block indices
    row, col = row.reshape(-1, B.nnz), col.reshape(-1, B.nnz)
    row += B.row
    col += B.col
    row, col = row.ravel(), col.ravel()

    # compute block entries
    data = data.reshape(-1, B.nnz) * B.data
    data = data.ravel()

    return coo.coo_matrix((data, (row, col)), shape=out_shape).asformat(format)
