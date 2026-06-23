from __future__ import annotations

import numpy
import cupy
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils


def _any_sparray(*args):
    """Return True if any argument is a sparse array (not matrix)."""
    return any(isinstance(a, _base.sparray) for a in args
               if _base.issparse(a))


def matrix_transpose(A):
    """Transpose the last two axes of a sparse object.

    Args:
        A (cupyx.scipy.sparse): Sparse object (CuPy currently supports
            2-D only).

    Returns:
        cupyx.scipy.sparse: ``A`` with its last two axes swapped (same
        array vs matrix type as the input).

    .. seealso:: :func:`scipy.sparse.matrix_transpose`
    """
    if not _base.issparse(A):
        raise TypeError('matrix_transpose expected a sparse object')
    return A.transpose()


def swapaxes(A, axis1, axis2):
    """Interchange two axes of a sparse object.

    Args:
        A (cupyx.scipy.sparse): Sparse 2-D object.
        axis1 (int): First axis (in ``[-2, 1]``).
        axis2 (int): Second axis (in ``[-2, 1]``).

    Returns:
        cupyx.scipy.sparse: ``A`` with axes swapped.

    .. seealso:: :func:`scipy.sparse.swapaxes`
    """
    if not _base.issparse(A):
        raise TypeError('swapaxes expected a sparse object')
    a1 = axis1 + 2 if axis1 < 0 else axis1
    a2 = axis2 + 2 if axis2 < 0 else axis2
    if a1 not in (0, 1) or a2 not in (0, 1):
        raise ValueError('axis out of range for 2-D sparse object')
    if a1 == a2:
        return A.copy()
    return A.transpose()


def permute_dims(A, axes=None, copy=False):
    """Permute the axes of a sparse object.

    Args:
        A (cupyx.scipy.sparse): Sparse 2-D object.
        axes (tuple of int or None): Permutation of axis indices.
            Defaults to reversing the axes (i.e. transpose).
        copy (bool): If ``True``, the result does not share data with
            ``A``.

    Returns:
        cupyx.scipy.sparse: ``A`` with axes permuted.

    .. seealso:: :func:`scipy.sparse.permute_dims`
    """
    if not _base.issparse(A):
        raise TypeError('permute_dims expected a sparse object')
    if axes is None:
        axes = (1, 0)
    axes = tuple(a + 2 if a < 0 else a for a in axes)
    if sorted(axes) != [0, 1]:
        raise ValueError('axes must be a permutation of (0, 1)')
    if axes == (0, 1):
        return A.copy() if copy else A
    return A.transpose(copy=copy)


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
            idx_dtype = _sputils.get_index_dtype(maxval=max(m, n))
            indptr = cupy.arange(n + 1, dtype=idx_dtype)
            indices = cupy.arange(n, dtype=idx_dtype)
            data = cupy.ones(n, dtype=dtype)
            if format == 'csr':
                cls = _csr.csr_matrix
            else:
                cls = _csc.csc_matrix
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            idx_dtype = _sputils.get_index_dtype(maxval=max(m, n))
            row = cupy.arange(n, dtype=idx_dtype)
            col = cupy.arange(n, dtype=idx_dtype)
            data = cupy.ones(n, dtype=dtype)
            return _coo.coo_matrix((data, (row, col)), (n, n))

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
    return _dia.dia_matrix((data, diags), shape=(m, n)).asformat(format)


def _compressed_sparse_stack(blocks, axis):
    """Fast path for stacking CSR/CSC matrices
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = cupy.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    all_idx = [b.indptr for b in blocks] + [b.indices for b in blocks]
    idx_dtype = _sputils.get_index_dtype(arrays=all_idx,
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
    use_array = _any_sparray(*blocks)
    if axis == 0:
        cls = _csr.csr_array if use_array else _csr.csr_matrix
        shape = (sum_dim, constant_dim)
    else:
        cls = _csc.csc_array if use_array else _csc.csc_matrix
        shape = (constant_dim, sum_dim)
    return cls._from_parts(
        data, indices, indptr, shape)


def hstack(blocks, format=None, dtype=None):
    """Stacks sparse arrays/matrices horizontally (column wise).

    Args:
        blocks (sequence of cupyx.scipy.sparse): sparse objects to stack.
        format (str): sparse format of the result (e.g. ``'csr'``).  By
            default an appropriate sparse format is returned.  This choice
            is subject to change.
        dtype (dtype, optional): The data-type of the output.  If not
            given, the dtype is determined from that of ``blocks``.

    Returns:
        cupyx.scipy.sparse: The stacked sparse object.  Returns a sparse
        array when *any* input is a sparse array, else a sparse matrix
        (matches scipy).

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
        return _coo.coo_matrix((0, 0), dtype=dtype)

    # check for fast path cases
    if (N == 1 and format in (None, 'csr') and
            all(_base.issparse(b) and b.format == 'csr'
                for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 0)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif (M == 1 and format in (None, 'csc')
          and all(_base.issparse(b) and b.format == 'csc'
                  for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 1)
        if dtype is not None:
            A = A.astype(dtype)
        return A

    block_mask = numpy.zeros((M, N), dtype=bool)
    brow_lengths = numpy.zeros(M+1, dtype=numpy.int64)
    bcol_lengths = numpy.zeros(N+1, dtype=numpy.int64)

    # Detect array type before COO conversion loses it
    _use_array = _any_sparray(*blocks_flat)

    # Check if any input block has int64 indices before conversion
    # to COO (the COO constructor may downcast via check_contents).
    # Each format stores indices in a different attribute: CSR/CSC use
    # ``indices``, COO uses ``row``, DIA uses ``offsets``.
    def _block_index_dtype(b):
        for name in ('indices', 'row', 'offsets'):
            arr = getattr(b, name, None)
            if arr is not None:
                return arr.dtype
        return None

    _has_int64 = any(
        _block_index_dtype(b) == cupy.int64 for b in blocks_flat)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                A = _coo.coo_matrix(blocks[i][j])
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

    # Rebuild blocks_flat after COO conversion so that .nnz and
    # .dtype are available for dense inputs that were converted.
    blocks_flat = [blocks[i][j] for i in range(M) for j in range(N)
                   if blocks[i][j] is not None]
    nnz = sum(block.nnz for block in blocks_flat)
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks_flat]
        dtype = _sputils.upcast(*all_dtypes) if all_dtypes else None

    row_offsets = numpy.cumsum(brow_lengths)
    col_offsets = numpy.cumsum(bcol_lengths)

    shape = (row_offsets[-1], col_offsets[-1])

    data = cupy.empty(nnz, dtype=dtype)
    # Propagate int64 from input blocks: if any input block had int64
    # indices (checked before COO conversion), the output should too.
    if _has_int64:
        idx_dtype = numpy.int64
    else:
        idx_dtype = _sputils.get_index_dtype(
            maxval=max(int(shape[0]), int(shape[1])) - 1)
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

    coo_cls = _coo.coo_array if _use_array else _coo.coo_matrix
    A = coo_cls._from_parts(data, row, col, shape)
    A.has_canonical_format = False
    return A.asformat(format)


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

    k = int(density * m * n)

    if random_state is None:
        random_state = cupy.random
    elif isinstance(random_state, (int, cupy.integer)):
        random_state = cupy.random.RandomState(random_state)

    if data_rvs is None:
        data_rvs = random_state.rand

    mn = m * n

    tp = numpy.int64 if mn > numpy.iinfo(numpy.int32).max \
        else numpy.int32
    ind = random_state.choice(mn, size=k, replace=False)
    ind = ind.astype(tp, copy=False)
    j = ind // m
    i = ind - j * m
    vals = data_rvs(k).astype(dtype)
    return _coo.coo_matrix(
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
    if _sputils.isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or _sputils.isscalarlike(diagonals[0]):
            diagonals = [cupy.atleast_1d(diagonals)]
        else:
            raise ValueError('Different number of diagonals and offsets.')
    else:
        diagonals = list(map(cupy.atleast_1d, diagonals))

    if isinstance(offsets, cupy.ndarray):
        offsets = offsets.get()  # synchronize!
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

    return _dia.dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)


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

    use_array = _any_sparray(A, B)
    coo_cls = _coo.coo_array if use_array else _coo.coo_matrix
    # Use the array path on input conversion when any input is a sparse
    # array so int64 indices survive the trip through COO.
    A = coo_cls(A)
    B = coo_cls(B)
    out_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        return coo_cls(out_shape).asformat(format)

    # Choose the output index dtype.
    #   - Sparray path: ``_get_index_dtype`` is called from a sparray
    #     instance so ``check_contents`` is forced off; the input
    #     arrays' dtypes are preserved (so kron(int64, int64) stays
    #     int64 even when the output shape would also fit int32).
    #     Mirrors scipy 1.17.
    #   - Matrix path: the legacy minimum-required policy (int32 unless
    #     the output shape forces int64).  Matches scipy matrix.
    if use_array:
        dtype = A._get_index_dtype(
            (A.row, A.col, B.row, B.col), maxval=max(out_shape))
    elif max(out_shape[0], out_shape[1]) > cupy.iinfo('int32').max:
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

    return coo_cls(
        (data, (row, col)), shape=out_shape).asformat(format)


def kronsum(A, B, format=None):
    """Kronecker sum of sparse matrices A and B.

    Kronecker sum is the sum of two Kronecker products
    ``kron(I_n, A) + kron(B, I_m)``, where ``I_n`` and ``I_m`` are identity
    matrices.

    Args:
        A (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        B (cupyx.scipy.sparse.spmatrix): a sparse matrix.
        format (str): the format of the returned sparse matrix.

    Returns:
        cupyx.scipy.sparse.spmatrix:
            Generated sparse matrix with the specified ``format``.

    .. seealso:: :func:`scipy.sparse.kronsum`

    """
    use_array = _any_sparray(A, B)
    src_cls = _coo.coo_array if use_array else _coo.coo_matrix
    A = src_cls(A)
    B = src_cls(B)

    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square matrix')

    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square matrix')

    dtype = _sputils.upcast(A.dtype, B.dtype)

    L = kron(eye(B.shape[0], dtype=dtype), A, format=format)
    R = kron(B, eye(A.shape[0], dtype=dtype), format=format)

    return (L + R).asformat(format)


# --- Array-returning construction functions ---

_array_containers = {
    'csr': lambda: _csr.csr_array,
    'csc': lambda: _csc.csc_array,
    'coo': lambda: _coo.coo_array,
    'dia': lambda: _dia.dia_array,
}


def _to_array(matrix, format=None):
    """Convert a sparse matrix result to the matching array type.

    When the requested format has no implemented array-side conversion
    (e.g. CSR -> DIA), the result is returned as ``csr_array``.
    """
    fmt = format or matrix.format
    cls = _array_containers.get(fmt, lambda: _csr.csr_array)()
    if isinstance(matrix, cls):
        return matrix
    arr = _csr.csr_array(matrix)
    if fmt and fmt != 'csr':
        try:
            arr = arr.asformat(fmt)
        except NotImplementedError:
            pass
    return arr


def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Creates a sparse array with ones on diagonal.

    Args:
        m (int): Number of rows.
        n (int or None): Number of columns. Defaults to ``m``.
        k (int): Diagonal to place ones on.
        dtype: Type of array to create.
        format (str or None): Format of the result.

    Returns:
        cupyx.scipy.sparse.sparray

    .. seealso:: :func:`scipy.sparse.eye_array`

    """
    if n is None:
        n = m
    m, n = int(m), int(n)
    return _to_array(eye(m, n, k=k, dtype=dtype, format=format), format)


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None,
                dtype=None):
    """Construct a sparse array from diagonals.

    Args:
        diagonals: Array of diagonal values.
        offsets (int or sequence of int): Diagonals to set.
        shape (tuple or None): Shape of the result.
        format (str or None): Sparse format of the result.
        dtype: Data type of the result.

    Returns:
        cupyx.scipy.sparse.sparray

    .. seealso:: :func:`scipy.sparse.diags_array`

    """
    return _to_array(
        diags(diagonals, offsets=offsets, shape=shape, format=format,
              dtype=dtype), format)


def block_array(blocks, *, format=None, dtype=None):
    """Build a sparse array from sparse sub-blocks.

    This is the array-returning equivalent of :func:`bmat`: even when none
    of the input ``blocks`` are sparse arrays the result is still a
    sparse array.

    Args:
        blocks (array_like): Grid of sparse arrays/matrices with compatible
            shapes.  An entry of ``None`` denotes an all-zero block.
        format (str, optional): Sparse format of the result (e.g. ``'csr'``).
            By default, an appropriate format is chosen.
        dtype (dtype, optional): Data type of the output.  Defaults to a
            promotion across input dtypes.

    Returns:
        cupyx.scipy.sparse.sparray: Stacked sparse array.

    .. seealso:: :func:`scipy.sparse.block_array`, :func:`bmat`
    """
    result = bmat(blocks, format=format, dtype=dtype)
    # Preserve sparse-array-ness even when all inputs were matrices/dense.
    if isinstance(result, _base.sparray):
        return result
    return _to_array(result, format)


def block_diag(mats, format=None, dtype=None):
    """Build a block-diagonal sparse object from a sequence of blocks.

    Args:
        mats (sequence): Input sparse arrays/matrices, dense ndarrays,
            scalars, or Python lists/tuples.  Each block becomes a
            diagonal sub-block in the output.
        format (str, optional): Sparse format of the result.  Defaults to
            ``'coo'`` when not specified.
        dtype (dtype, optional): Data type of the output.

    Returns:
        cupyx.scipy.sparse: Sparse array if any input is a sparse array,
        else sparse matrix (matching scipy).

    .. seealso:: :func:`scipy.sparse.block_diag`
    """
    use_array = any(isinstance(a, _base.sparray) for a in mats)
    coo_cls = _coo.coo_array if use_array else _coo.coo_matrix

    def _normalize_dense(a):
        """Promote a dense input (list/tuple/scalar/ndarray) to a 2-D
        ``cupy.ndarray`` with a sparse-supported dtype.

        Integer-typed dense input is upcast to float64 since cuSPARSE
        won't store it.  This matches scipy's behaviour for integer
        ``diags`` input -- the upstream FutureWarning is filtered out
        in ``pyproject.toml``.
        """
        a = cupy.atleast_2d(cupy.asarray(a))
        if not _sputils.is_sparse_data_dtype(a.dtype):
            a = a.astype(cupy.float64, copy=False)
        return a

    rows = []
    cols = []
    datas = []
    # Collect int64 coord arrays so ``get_index_dtype`` picks int64 for
    # the assembled output when any input block needs it.  Each coord
    # is checked independently because a ``tocoo()`` implementation
    # could drop one side to int32.
    idx_arrays = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        if _base.issparse(a):
            a_coo = a.tocoo()
            if a_coo.row.dtype == cupy.int64:
                idx_arrays.append(a_coo.row)
            if a_coo.col.dtype == cupy.int64:
                idx_arrays.append(a_coo.col)
            nrows, ncols = a_coo.shape
            rows.append(a_coo.row + r_idx)
            cols.append(a_coo.col + c_idx)
            datas.append(a_coo.data)
        else:
            ad = _normalize_dense(a)
            nrows, ncols = ad.shape
            r, c = cupy.divmod(
                cupy.arange(nrows * ncols), ncols)
            rows.append(r + r_idx)
            cols.append(c + c_idx)
            datas.append(ad.ravel())
        r_idx += nrows
        c_idx += ncols

    idx_dtype = _sputils.get_index_dtype(
        arrays=tuple(idx_arrays),
        maxval=max(r_idx, c_idx) if r_idx or c_idx else None)
    if rows:
        row = cupy.concatenate(rows).astype(idx_dtype, copy=False)
        col = cupy.concatenate(cols).astype(idx_dtype, copy=False)
        data = cupy.concatenate(datas)
    else:
        # Empty input: build an explicit (0, 0) sparse output.
        row = cupy.zeros(0, dtype=idx_dtype)
        col = cupy.zeros(0, dtype=idx_dtype)
        data = cupy.zeros(0, dtype=dtype if dtype is not None else 'd')
    new_shape = (r_idx, c_idx)
    out = coo_cls._from_parts(data, row, col, new_shape)
    if dtype is not None:
        out = out.astype(dtype)
    return out.asformat(format)


def random_array(shape, *, density=0.01, format='coo', dtype=None,
                 rng=None, data_sampler=None):
    """Generate a sparse random array.

    Args:
        shape (tuple): Shape of the array (m, n).
        density (float): Density of generated values.
        format (str): Sparse format of the result.
        dtype: Data type of generated values.
        rng: Random number generator (numpy or cupy).
        data_sampler: Function that accepts a size argument.

    Returns:
        cupyx.scipy.sparse.sparray

    .. seealso:: :func:`scipy.sparse.random_array`

    """
    m, n = shape
    return _to_array(
        random(m, n, density=density, format=format, dtype=dtype,
               random_state=rng, data_rvs=data_sampler), format)
