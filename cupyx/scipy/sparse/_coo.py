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
        if shape is not None and len(shape) not in self._allow_nd:
            raise ValueError(
                'Only two-dimensional sparse arrays are supported.'
                if 1 not in self._allow_nd
                else 'Only 1-D and 2-D sparse arrays are supported.')
        if shape is not None:
            # Catch negative dimensions before the index-bounds checks
            # below would otherwise fire with a misleading
            # "column index exceeds matrix dimensions" message.
            shape = _util.check_shape(shape, allow_nd=self._allow_nd)
        # A 1-D array is stored as a (1, N) row vector (``row`` all-zeros,
        # ``col`` the coordinate) presenting ``shape == (N,)``; see
        # ``_as_2d``.  ``is_1d`` is reconciled from ``shape`` below.
        is_1d = False
        # Number of coordinate arrays supplied (tuple-of-coords form only);
        # validated against the shape's ndim below.
        n_coords = None

        if _base.issparse(arg1):
            # A whole sparse input fixes the dimensionality; a conflicting
            # explicit shape must not silently reinterpret it (would corrupt
            # a 2-D input presented as 1-D, or vice versa).
            _util.check_input_ndim(shape, arg1.ndim)
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

        elif _util.isshape(arg1, allow_nd=self._allow_nd):
            # ``isshape`` is a pure type-check; ``check_shape`` raises
            # ``ValueError("'shape' elements cannot be negative")`` on a
            # negative dimension to match scipy's message.
            shape = _util.check_shape(arg1, allow_nd=self._allow_nd)
            data = cupy.zeros(0, dtype if dtype else 'd')
            idx_dtype = _sputils.get_index_dtype(maxval=max(shape))
            row = cupy.zeros(0, dtype=idx_dtype)
            col = cupy.zeros(0, dtype=idx_dtype)
            # shape and copy argument is ignored
            copy = False

            self.has_canonical_format = True

        elif _scipy_available and scipy.sparse.issparse(arg1):
            _util.check_input_ndim(shape, arg1.ndim)
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
                data, coords = arg1
            except (TypeError, ValueError):
                raise TypeError('invalid input format')
            # ``coords`` is a sequence of per-dimension index arrays:
            # ``(col,)`` (1-D) or ``(row, col)`` (2-D).  A single dense
            # ``(ndim, nnz)`` array is also accepted -- its rows are the
            # per-dimension coordinates -- matching scipy and the former
            # ``data, (row, col) = arg1`` unpacking.  A bare 1-D dense
            # array is not a valid coordinate sequence.
            if _base.isdense(coords) and coords.ndim != 2:
                raise TypeError('invalid input format')
            try:
                coords = tuple(coords)
            except TypeError:
                raise TypeError('invalid input format')

            n_coords = len(coords)
            if len(coords) == 1 and 1 in self._allow_nd:
                (col,) = coords
                if not (_base.isdense(data) and data.ndim == 1 and
                        _base.isdense(col) and col.ndim == 1):
                    raise ValueError(
                        'coordinate and data arrays must be 1-D')
                if len(data) != len(col):
                    raise ValueError(
                        'coordinate and data array must all be the '
                        'same length')
                row = cupy.zeros_like(col)
                is_1d = True
            elif len(coords) == 2:
                row, col = coords
                if not (_base.isdense(data) and data.ndim == 1 and
                        _base.isdense(row) and row.ndim == 1 and
                        _base.isdense(col) and col.ndim == 1):
                    raise ValueError(
                        'row, column, and data arrays must be 1-D')
                if not (len(data) == len(row) == len(col)):
                    raise ValueError(
                        'row, column, and data array must all be the '
                        'same length')
            else:
                raise ValueError('invalid number of coordinate arrays')

            self.has_canonical_format = False

        elif _base.isdense(arg1):
            if arg1.ndim > 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            if arg1.ndim == 0 and isinstance(self, _base.sparray):
                # scipy rejects scalar (0-D) input for sparse arrays.
                raise TypeError(
                    f'{type(self).__name__} does not support scalar (0-D) '
                    'input; provide a 1-D or 2-D array')
            if arg1.ndim >= 1:
                # A dense input fixes the dimensionality; reject a shape of a
                # different ndim (0-D matrix promotion is handled below).
                _util.check_input_ndim(shape, arg1.ndim)
            if arg1.ndim == 1 and 1 in self._allow_nd:
                # 1-D dense input -> 1-D coo_array.
                dense = cupy.atleast_1d(arg1)
                col = dense.nonzero()[0]
                data = dense[col]
                row = cupy.zeros_like(col)
                shape = dense.shape
            else:
                dense = cupy.atleast_2d(arg1)
                row, col = dense.nonzero()
                data = dense[row, col]
                shape = dense.shape

            self.has_canonical_format = True

        else:
            raise TypeError('invalid input format')

        # A tuple-of-coords input must supply exactly one coordinate array
        # per dimension of the requested shape (scipy parity); otherwise the
        # 1-D backing would silently keep a bogus ``row`` or mis-dimension.
        if (n_coords is not None and shape is not None
                and len(shape) != n_coords):
            raise ValueError(
                f'mismatching number of index arrays for shape; got '
                f'{n_coords}, expected {len(shape)}')
        # When the shape is known, it is authoritative for dimensionality;
        # otherwise ``is_1d`` was set by the (data, (col,)) branch above.
        if shape is not None:
            is_1d = len(shape) == 1

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
            if is_1d:
                # Only ``col`` carries information for a 1-D array.
                bounds = cupy.stack((col.max(), col.min())).get()  # sync!
                cmax, cmin = (int(b) for b in bounds)
                if shape is None:
                    shape = (cmax + 1,)
                if cmax >= shape[0]:
                    raise ValueError('column index exceeds matrix dimensions')
                if cmin < 0:
                    raise ValueError('negative column index found')
            else:
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
        self._shape = _util.check_shape(shape, allow_nd=self._allow_nd)

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
        shape = _util.check_shape(shape, allow_nd=cls._allow_nd)
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
        # ``row``/``col`` hold coordinates bounded by ``max(shape)``.
        # Unlike CSR/CSC there is no cumulative ``indptr``, so
        # ``max(shape)`` -- not ``prod(shape)`` or ``nnz`` -- is the only
        # dtype constraint.  (Can't fold into ``check_shape``: it depends
        # on the array dtype, not just the shape.)
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
        """Tuple of coordinate arrays.

        For a 2-D array this is ``(row, col)``; for a 1-D array it is the
        single coordinate ``(col,)`` (``row`` is an all-zeros backing).
        Mirrors :attr:`scipy.sparse.coo_array.coords`.
        """
        if self.ndim == 1:
            return (self.col,)
        return (self.row, self.col)

    @coords.setter
    def coords(self, value):
        if not isinstance(value, tuple) or len(value) != self.ndim:
            raise ValueError(
                f'coords must be a {self.ndim}-tuple of arrays')
        if self.ndim == 1:
            (self.col,) = value
            self.row = cupy.zeros_like(self.col)
        else:
            self.row, self.col = value

    def _as_2d(self):
        """Return the ``(1, N)`` 2-D backing of a 1-D array (else self).

        A 1-D COO array stores ``row`` as an all-zeros vector and ``col``
        as the coordinate, so the backing is built directly from the
        existing arrays with shape :attr:`_shape_as_2d`.
        """
        if self.ndim != 1:
            return self
        return type(self)._from_parts(
            self.data, self.row, self.col, self._shape_as_2d,
            has_canonical_format=self.has_canonical_format)

    def diagonal(self, k=0):
        """Returns the k-th diagonal of the matrix.

        Args:
            k (int, optional): Which diagonal to get, corresponding to elements
            a[i, i+k]. Default: 0 (the main diagonal).

        Returns:
            cupy.ndarray : The k-th diagonal.
        """
        self._require_2d('diagonal')
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
            values: New values of the diagonal elements.  Accepts a
                scalar, list, or 1-D array; any non-cupy input is
                coerced via :func:`cupy.asarray`.  Values may have any
                length: if longer than the diagonal, extras are
                ignored; if shorter, remaining diagonal entries are
                left unchanged.  A scalar broadcasts to the whole
                diagonal.
            k (int, optional): Which off-diagonal to set, corresponding to
                elements a[i,i+k]. Default: 0 (the main diagonal).

        """
        self._require_2d('setdiag')
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
        if self.ndim == 1:
            # The only axis collapses to a scalar count.
            _sputils.validate_axis_1d(axis)
            return int(cupy.count_nonzero(self.data))
        if axis is None:
            return int(cupy.count_nonzero(self.data))
        if axis < 0:
            axis += 2
        if axis < 0 or axis >= 2:
            raise ValueError('axis out of bounds')
        out_dim = self.shape[1 - axis]
        mask = self.data != 0
        coord = (self.col if axis == 0 else self.row)[mask]
        # Nothing left to count (empty matrix or all explicit zeros):
        # scipy returns the zero-filled axis vector.
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
        if isinstance(self, _base.sparray):
            sp_cls = scipy.sparse.coo_array
        else:
            sp_cls = scipy.sparse.coo_matrix
        if self.ndim == 1:
            # 1-D coo_array (scipy >= 1.13): single coordinate array.
            col = self.col.get(stream)
            return sp_cls((data, (col,)), shape=self.shape)
        row = self.row.get(stream)
        col = self.col.get(stream)
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

        shape = _sputils.check_shape(
            shape, self.shape, allow_nd=self._allow_nd)

        if shape == self.shape:
            return self

        # Map both source and target through their (1, N) 2-D backing so
        # the flatten/unravel math below handles 1-D arrays uniformly.
        nrows, ncols = self._shape_as_2d
        tgt_nrows, tgt_ncols = (1, shape[0]) if len(shape) == 1 else shape

        if order == 'C':  # C to represent matrix in row major format
            dtype = _sputils.get_index_dtype(
                maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))
            flat_indices = cupy.multiply(ncols, self.row,
                                         dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, tgt_ncols)
        elif order == 'F':  # column-major: flat = col * nrows + row
            dtype = _sputils.get_index_dtype(
                maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))
            flat_indices = cupy.multiply(nrows, self.col,
                                         dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, tgt_nrows)
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
        if self.ndim == 1:
            # Densify through the (1, N) backing (correct duplicate and
            # explicit-zero handling), then drop the length-1 axis.
            dense = self._as_2d().tocsr().toarray(order=order)
            result = dense.reshape(self.shape)
            if out is not None:
                _core.elementwise_copy(result, out)
                return out
            return result
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
        if self.ndim == 1:
            raise ValueError(
                'Cannot convert a 1-D sparse array to csc format')

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
        if self.ndim == 1:
            # Convert the (1, N) backing, then present the result as 1-D.
            csr2d = self._as_2d().tocsr(copy=copy)
            return self._csr_container._from_parts(
                csr2d.data, csr2d.indices, csr2d.indptr, self.shape,
                has_canonical_format=getattr(
                    csr2d, '_has_canonical_format', None),
                has_sorted_indices=getattr(
                    csr2d, '_has_sorted_indices', None))

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
        if self.ndim == 1:
            # Transpose of a 1-D array is a no-op (matches scipy).
            return self.copy() if copy else self
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

    Unlike :class:`coo_matrix`, this supports 1-D shapes (``ndim == 1``)
    in addition to 2-D, matching :class:`scipy.sparse.coo_array`.  A 1-D
    array of length ``N`` is stored as a ``(1, N)`` row vector
    (``row`` all-zeros, ``col`` the coordinate) and presents
    ``shape == (N,)``; see :attr:`_spbase._shape_as_2d`.

    .. seealso:: :class:`scipy.sparse.coo_array`
    """

    _allow_nd = (1, 2)


def isspmatrix_coo(x):
    """Checks if a given matrix is of COO format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.coo_matrix`.

    """
    return isinstance(x, coo_matrix)
