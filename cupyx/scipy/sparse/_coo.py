from __future__ import annotations

import numpy
try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import _core
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils


class _coo_base(sparse_data._data_matrix):
    """COO format base (shared by ``coo_matrix`` and ``coo_array``).

    This can be instantiated in several ways:

    ``coo_*(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``coo_*(S)``
        ``S`` is another sparse object.  Equivalent to ``S.tocoo()``.
    ``coo_*((M, N), [dtype])``
        Constructs an empty (M, N)-shaped sparse object.  Default dtype
        is float64.
    ``coo_*((data, (row, col)), shape=...)``
        ``data``, ``row`` and ``col`` are 1-D :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape; must be a 2-tuple of ints.
        dtype: Data type; must be representable as :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given data are always used.

    .. seealso::
       :class:`scipy.sparse.coo_array`,
       :class:`scipy.sparse.coo_matrix`
    """

    format = 'coo'

    _sum_duplicates_diff = _core.ElementwiseKernel(
        'raw T row, raw T col',
        'T diff',
        '''
        T diff_out = 1;
        if (i == 0 || row[i - 1] == row[i] && col[i - 1] == col[i]) {
          diff_out = 0;
        }
        diff = diff_out;
        ''', 'cupyx_scipy_sparse_coo_sum_duplicates_diff')

    def __init__(self, arg1, shape=None, dtype=None, copy=False,
                 *, maxprint=None):
        if maxprint is not None:
            self.maxprint = maxprint
        if shape is not None and len(shape) != 2:
            raise ValueError(
                'Only two-dimensional sparse arrays are supported.')
        if shape is not None:
            # Catch negative dimensions before the index-bounds checks
            # below would otherwise fire with a misleading
            # "column index exceeds matrix dimensions" message.
            shape = _util.check_shape(shape)

        if _base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            row = x.row
            col = x.col

            if arg1.format != self.format:
                # When formats are different, all arrays are already copied
                copy = False

            if shape is None:
                shape = arg1.shape

            self.has_canonical_format = x.has_canonical_format

        elif _util.isshape(arg1):
            # ``isshape`` is a pure type-check; ``check_shape`` raises
            # ``ValueError("'shape' elements cannot be negative")`` on a
            # negative dimension to match scipy's message.
            m, n = _util.check_shape(arg1)
            data = cupy.zeros(0, dtype if dtype else 'd')
            idx_dtype = _sputils.get_index_dtype(maxval=max(m, n))
            row = cupy.zeros(0, dtype=idx_dtype)
            col = cupy.zeros(0, dtype=idx_dtype)
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False

            self.has_canonical_format = True

        elif _scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse.
            # Preserve scipy's index dtype (scipy uses
            # get_index_dtype internally).
            x = arg1.tocoo()
            data = cupy.array(x.data)
            row = cupy.array(x.row, dtype=x.row.dtype)
            col = cupy.array(x.col, dtype=x.col.dtype)
            copy = False
            if shape is None:
                shape = arg1.shape

            self.has_canonical_format = x.has_canonical_format

        elif isinstance(arg1, tuple) and len(arg1) == 2:
            try:
                data, (row, col) = arg1
            except (TypeError, ValueError):
                raise TypeError('invalid input format')

            if not (_base.isdense(data) and data.ndim == 1 and
                    _base.isdense(row) and row.ndim == 1 and
                    _base.isdense(col) and col.ndim == 1):
                raise ValueError('row, column, and data arrays must be 1-D')
            if not (len(data) == len(row) == len(col)):
                raise ValueError(
                    'row, column, and data array must all be the same length')

            self.has_canonical_format = False

        elif _base.isdense(arg1):
            if arg1.ndim > 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            dense = cupy.atleast_2d(arg1)
            row, col = dense.nonzero()
            data = dense[row, col]
            shape = dense.shape

            self.has_canonical_format = True

        else:
            raise TypeError('invalid input format')

        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)

        if not _sputils.is_sparse_data_dtype(dtype):
            raise ValueError(
                'Only bool, float32, float64, complex64 and complex128'
                ' are supported')

        data = data.astype(dtype, copy=copy)
        # Choose index dtype: int32 when values fit, int64 when they don't.
        # For matrices, check_contents=True may downcast int64 to int32.
        # For arrays, _get_index_dtype disables check_contents so user
        # dtypes are preserved.
        if shape is not None:
            maxval = max(shape)
        else:
            maxval = None
        idx_dtype = self._get_index_dtype(
            (row, col), maxval=maxval, check_contents=True)
        row = row.astype(idx_dtype, copy=copy)
        col = col.astype(idx_dtype, copy=copy)

        if shape is None and (len(row) == 0 or len(col) == 0):
            raise ValueError(
                'cannot infer dimensions from zero sized index arrays')

        if len(data) > 0:
            # Fuse the four max/min reductions into one D2H read so
            # the constructor syncs once instead of up to six times.
            bounds = cupy.stack(
                (row.max(), col.max(), row.min(), col.min())
            ).get()  # synchronize!
            rmax, cmax, rmin, cmin = (int(b) for b in bounds)
            if shape is None:
                shape = (rmax + 1, cmax + 1)
            if rmax >= shape[0]:
                raise ValueError('row index exceeds matrix dimensions')
            if cmax >= shape[1]:
                raise ValueError('column index exceeds matrix dimensions')
            if rmin < 0:
                raise ValueError('negative row index found')
            if cmin < 0:
                raise ValueError('negative column index found')

        sparse_data._data_matrix.__init__(self, data)
        self.row = row
        self.col = col
        self._shape = _util.check_shape(shape)

    @classmethod
    def _from_parts(cls, data, row, col, shape,
                    has_canonical_format=False):
        """Construct from pre-validated arrays (no check_contents).

        Internal API for building COO matrices when the caller has
        already determined the correct index dtype.  Skips the
        check_contents=True downcast that the tuple-2 constructor
        applies.

        Args:
            has_canonical_format (bool): Defaults to ``False`` (not
                known to be canonical).

        Raises:
            ValueError: If ``row`` and ``col`` dtypes differ, the
                arrays' lengths are inconsistent, ``shape`` contains
                a negative dimension, or the index dtype is too
                narrow to address ``shape``.
        """
        shape = _util.check_shape(shape)
        if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
            raise ValueError(
                f'data, row, and col must be 1-D, got ndim '
                f'{data.ndim}, {row.ndim}, {col.ndim}')
        if row.dtype != col.dtype:
            raise ValueError(
                f'row and col must have the same dtype, '
                f'got {row.dtype} and {col.dtype}')
        if data.size != row.size or data.size != col.size:
            raise ValueError(
                f'data, row, and col must have the same length, '
                f'got {data.size}, {row.size}, and {col.size}')
        if max(shape) > numpy.iinfo(row.dtype).max:
            raise ValueError(
                f'shape {shape} too large for index dtype {row.dtype}')
        A = cls.__new__(cls)
        sparse_data._data_matrix.__init__(A, data)
        A.row = row
        A.col = col
        A._shape = shape
        A.has_canonical_format = has_canonical_format
        return A

    def _with_data(self, data, copy=True):
        """Return a matrix with the same sparsity structure but
        different data.  Preserves has_canonical_format.
        """
        return type(self)._from_parts(
            data,
            self.row.copy() if copy else self.row,
            self.col.copy() if copy else self.col,
            self.shape,
            has_canonical_format=self.has_canonical_format)

    @property
    def coords(self):
        """Tuple of coordinate arrays ``(row, col)``.

        Mirrors :attr:`scipy.sparse.coo_array.coords` (CuPy is 2-D only).
        """
        return (self.row, self.col)

    @coords.setter
    def coords(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(
                'coords must be a 2-tuple of arrays for 2-D sparse')
        self.row, self.col = value

    def diagonal(self, k=0):
        """Returns the k-th diagonal of the matrix.

        Args:
            k (int, optional): Which diagonal to get, corresponding to elements
            a[i, i+k]. Default: 0 (the main diagonal).

        Returns:
            cupy.ndarray : The k-th diagonal.
        """
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cupy.empty(0, dtype=self.data.dtype)
        diag = cupy.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                          dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            diag_coo = type(self)((self.data[diag_mask],
                                   (self.row[diag_mask], self.col[diag_mask])),
                                  shape=self.shape)
            diag_coo.sum_duplicates()
            row = diag_coo.row
            data = diag_coo.data
        diag[row + min(k, 0)] = data

        return diag

    def setdiag(self, values, k=0):
        """Set diagonal or off-diagonal elements of the array.

        Args:
            values (ndarray): New values of the diagonal elements. Values may
                have any length. If the diagonal is longer than values, then
                the remaining diagonal entries will not be set. If values are
                longer than the diagonal, then the remaining values are
                ignored. If a scalar value is given, all of the diagonal is set
                to it.
            k (int, optional): Which off-diagonal to set, corresponding to
                elements a[i,i+k]. Default: 0 (the main diagonal).

        """
        M, N = self.shape
        if (k > 0 and k >= N) or (k < 0 and -k >= M):
            raise ValueError("k exceeds matrix dimensions")
        # Coerce list/scalar/numpy input; matches scipy's
        # ``np.asarray(values)`` in ``_spbase.setdiag``.
        values = cupy.asarray(values, dtype=self.dtype)
        if values.ndim > 1:
            raise ValueError('values must be 0-d or 1-d')
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M + k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = cupy.logical_or(full_keep, self.col >= max_index)
            new_row = cupy.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = cupy.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N - k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = cupy.logical_or(full_keep, self.row >= max_index)
            new_row = cupy.arange(max_index, dtype=idx_dtype)
            new_col = cupy.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = cupy.full(max_index, values, dtype=self.dtype)

        # Update the internal structure.
        self.row = cupy.concatenate((self.row[keep], new_row))
        self.col = cupy.concatenate((self.col[keep], new_col))
        self.data = cupy.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        ind = self.data != 0
        self.data = self.data[ind]
        self.row = self.row[ind]
        self.col = self.col[ind]

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros."""
        if axis is None:
            return self.data.size
        else:
            raise ValueError

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Excludes explicit zeros.  Duplicates are summed first.

        Args:
            axis ({-2, -1, 0, 1, ``None``}):
                Count over the whole matrix, or along an axis.

        Returns:
            int or cupy.ndarray: Scalar count when ``axis=None``,
            otherwise a 1-D ``cupy.ndarray`` of length
            ``shape[1 - axis]``.
        """
        # Match scipy: dedup in place, then count.  COO is already in
        # row/col form so per-axis counts come straight from a bincount
        # on the appropriate coord array -- no format conversion needed.
        self.sum_duplicates()
        if axis is None:
            return int(cupy.count_nonzero(self.data))
        if axis < 0:
            axis += 2
        if axis < 0 or axis >= 2:
            raise ValueError('axis out of bounds')
        # ``cupy.bincount`` errors on empty input even with ``minlength``
        # (CUB max-reduction has no identity for zero-size arrays), so
        # short-circuit when nothing is stored or every entry is an
        # explicit zero.  scipy returns the zero-filled axis vector.
        out_dim = self.shape[1 - axis]
        if self.data.size == 0:
            return cupy.zeros(out_dim, dtype=cupy.intp)
        mask = self.data != 0
        coord = (self.col if axis == 0 else self.row)[mask]
        if coord.size == 0:
            return cupy.zeros(out_dim, dtype=cupy.intp)
        return cupy.bincount(
            coord.astype(cupy.int64), minlength=out_dim)

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.coo_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')

        data = self.data.get(stream)
        row = self.row.get(stream)
        col = self.col.get(stream)
        if isinstance(self, _base.sparray):
            sp_cls = scipy.sparse.coo_array
        else:
            sp_cls = scipy.sparse.coo_matrix
        return sp_cls((data, (row, col)), shape=self.shape)

    def reshape(self, *shape, order='C'):
        """Gives a new shape to a sparse matrix without changing its data.

        Args:
            shape (tuple):
                The new shape should be compatible with the original shape.
            order: {'C', 'F'} (optional)
                Read the elements using this index order. 'C' means to read and
                write the elements using C-like index order. 'F' means to read
                and write the elements using Fortran-like index order. Default:
                C.

        Returns:
            cupyx.scipy.sparse.coo_matrix: sparse matrix

        """

        shape = _sputils.check_shape(shape, self.shape)

        if shape == self.shape:
            return self

        nrows, ncols = self.shape

        if order == 'C':  # C to represent matrix in row major format
            dtype = _sputils.get_index_dtype(
                maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))
            flat_indices = cupy.multiply(ncols, self.row,
                                         dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            dtype = _sputils.get_index_dtype(
                maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))
            flat_indices = cupy.multiply(ncols, self.row,
                                         dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        return type(self)._from_parts(
            self.data, new_row, new_col, shape=shape)

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together.

        .. warning::
            When sorting the indices, CuPy follows the convention of cuSPARSE,
            which is different from that of SciPy. Therefore, the order of the
            output indices may differ:

            .. code-block:: python

                >>> #     1 0 0
                >>> # A = 1 1 0
                >>> #     1 1 1
                >>> data = cupy.array([1, 1, 1, 1, 1, 1], 'f')
                >>> row = cupy.array([0, 1, 1, 2, 2, 2], 'i')
                >>> col = cupy.array([0, 0, 1, 0, 1, 2], 'i')
                >>> A = cupyx.scipy.sparse.coo_matrix((data, (row, col)),
                ...                                   shape=(3, 3))
                >>> a = A.get()
                >>> A.sum_duplicates()
                >>> a.sum_duplicates()  # a is scipy.sparse.coo_matrix
                >>> A.row
                array([0, 1, 1, 2, 2, 2], dtype=int32)
                >>> a.row
                array([0, 1, 2, 1, 2, 2], dtype=int32)
                >>> A.col
                array([0, 0, 1, 0, 1, 2], dtype=int32)
                >>> a.col
                array([0, 0, 0, 1, 1, 2], dtype=int32)

        .. warning::
            Calling this function might synchronize the device.

        .. seealso::
           :meth:`scipy.sparse.coo_matrix.sum_duplicates`

        """
        if self.has_canonical_format:
            return
        # Note: The sorting order below follows the cuSPARSE convention (first
        # row then col, so-called row-major) and differs from that of SciPy, as
        # the cuSPARSE functions such as cusparseSpMV() assume this sorting
        # order.
        # See https://docs.nvidia.com/cuda/cusparse/index.html#coo-format
        keys = cupy.stack([self.col, self.row])
        order = cupy.lexsort(keys)
        src_data = self.data[order]
        src_row = self.row[order]
        src_col = self.col[order]
        diff = self._sum_duplicates_diff(src_row, src_col, size=self.row.size)

        if diff[1:].all():  # synchronize!
            # All elements have different indices.
            data = src_data
            row = src_row
            col = src_col
        else:
            # TODO(leofang): move the kernels outside this method
            # Use the actual index dtype (int64 when indices are
            # large).  ElementwiseKernel silently casts to the
            # declared type -- using 'int32' here would silently
            # truncate int64 index values > INT32_MAX.
            idx_dtype = self.row.dtype
            index = cupy.cumsum(diff, dtype=idx_dtype)
            size = int(index[-1]) + 1  # synchronize!
            data = cupy.zeros(size, dtype=self.data.dtype)
            row = cupy.empty(size, dtype=idx_dtype)
            col = cupy.empty(size, dtype=idx_dtype)
            if self.data.dtype.kind == 'b':
                cupy.ElementwiseKernel(
                    'T src_data, I src_row, I src_col, I index',
                    'raw T data, raw I row, raw I col',
                    '''
                    if (src_data) data[index] = true;
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'cupyx_scipy_sparse_coo_sum_duplicates_assign'
                )(src_data, src_row, src_col, index, data, row, col)
            elif self.data.dtype.kind == 'f':
                cupy.ElementwiseKernel(
                    'T src_data, I src_row, I src_col, I index',
                    'raw T data, raw I row, raw I col',
                    '''
                    atomicAdd(&data[index], src_data);
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'cupyx_scipy_sparse_coo_sum_duplicates_assign'
                )(src_data, src_row, src_col, index, data, row, col)
            elif self.data.dtype.kind == 'c':
                cupy.ElementwiseKernel(
                    'T src_real, T src_imag, I src_row, I src_col, '
                    'I index',
                    'raw T real, raw T imag, raw I row, raw I col',
                    '''
                    atomicAdd(&real[index], src_real);
                    atomicAdd(&imag[index], src_imag);
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'cupyx_scipy_sparse_coo_sum_duplicates_assign_complex'
                )(src_data.real, src_data.imag, src_row, src_col, index,
                  data.real, data.imag, row, col)

        self.data = data
        self.row = row
        self.col = col
        self.has_canonical_format = True

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order (str): Not supported.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same value.

        .. seealso:: :meth:`scipy.sparse.coo_matrix.toarray`

        """
        return self.tocsr().toarray(order=order, out=out)

    def tocoo(self, copy=False):
        """Converts the matrix to COOrdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupyx.scipy.sparse.coo_matrix: Converted matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in coo to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        from cupyx import cusparse

        if self.nnz == 0:
            idx = self.col.dtype
            n = self.shape[1]
            return self._csc_container._from_parts(
                cupy.empty(0, self.dtype),
                cupy.empty(0, idx),
                cupy.zeros(n + 1, idx),
                self.shape)
        # copy is silently ignored (in line with SciPy) because both
        # sum_duplicates and coosort change the underlying data
        x = self.copy()
        x.sum_duplicates()
        cusparse.coosort(x, 'c')
        result = cusparse.coo2csc(x)
        result.has_canonical_format = True
        if not isinstance(result, self._csc_container):
            result = self._csc_container(result)
        return result

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in coo to csr conversion.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        from cupyx import cusparse

        if self.nnz == 0:
            idx = self.row.dtype
            m = self.shape[0]
            return self._csr_container._from_parts(
                cupy.empty(0, self.dtype),
                cupy.empty(0, idx),
                cupy.zeros(m + 1, idx),
                self.shape)
        # copy is silently ignored (in line with SciPy) because both
        # sum_duplicates and coosort change the underlying data
        x = self.copy()
        x.sum_duplicates()
        cusparse.coosort(x, 'r')
        result = cusparse.coo2csr(x)
        result.has_canonical_format = True
        if not isinstance(result, self._csr_container):
            result = self._csr_container(result)
        return result

    def transpose(self, axes=None, copy=False):
        """Returns a transpose matrix.

        Args:
            axes: This option is not supported.
            copy (bool): If ``True``, a returned matrix shares no data.
                Otherwise, it shared data arrays as much as possible.

        Returns:
            cupyx.scipy.sparse.spmatrix: Transpose matrix.

        """
        if axes is not None:
            raise ValueError(
                'Sparse matrices do not support an \'axes\' parameter because '
                'swapping dimensions is the only logical permutation.')
        shape = self.shape[1], self.shape[0]
        if copy:
            data = self.data.copy()
            row, col = self.col.copy(), self.row.copy()
        else:
            data, row, col = self.data, self.col, self.row
        # Transposing swaps row/col, which generally destroys
        # canonical order (sorted by row then col).
        return type(self)._from_parts(
            data, row, col, shape,
            has_canonical_format=False)

    def dot(self, other):
        """Ordinary dot product"""
        if _util.isscalarlike(other):
            return type(self)._from_parts(
                self.data * other,
                self.row.copy(), self.col.copy(),
                self.shape,
                has_canonical_format=self.has_canonical_format)
        else:
            return self @ other


class coo_matrix(_base.spmatrix, _coo_base):
    """COOrdinate format sparse matrix.

    .. seealso:: :class:`scipy.sparse.coo_matrix`
    """
    pass


class coo_array(_coo_base, _base.sparray):
    """COOrdinate format sparse array.

    .. seealso:: :class:`scipy.sparse.coo_array`
    """
    pass


def isspmatrix_coo(x):
    """Checks if a given matrix is of COO format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.coo_matrix`.

    """
    return isinstance(x, coo_matrix)
