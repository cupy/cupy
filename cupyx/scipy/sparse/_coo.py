import numpy
try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import _core
from cupy import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils


class coo_matrix(sparse_data._data_matrix):

    """COOrdinate format sparse matrix.

    This can be instantiated in several ways.

    ``coo_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.

    ``coo_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocoo()``.

    ``coo_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.

    ``coo_matrix((data, (row, col)))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given data are always used.

    .. seealso::
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

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if shape is not None and len(shape) != 2:
            raise ValueError(
                'Only two-dimensional sparse arrays are supported.')

        if _base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            row = x.row
            col = x.col

            if arg1.format != self.format:
                # When formats are differnent, all arrays are already copied
                copy = False

            if shape is None:
                shape = arg1.shape

            self.has_canonical_format = x.has_canonical_format

        elif _util.isshape(arg1):
            m, n = arg1
            m, n = int(m), int(n)
            data = cupy.zeros(0, dtype if dtype else 'd')
            row = cupy.zeros(0, dtype='i')
            col = cupy.zeros(0, dtype='i')
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False

            self.has_canonical_format = True

        elif _scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse
            x = arg1.tocoo()
            data = cupy.array(x.data)
            row = cupy.array(x.row, dtype='i')
            col = cupy.array(x.col, dtype='i')
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

        if dtype != 'f' and dtype != 'd' and dtype != 'F' and dtype != 'D':
            raise ValueError(
                'Only float32, float64, complex64 and complex128'
                ' are supported')

        data = data.astype(dtype, copy=copy)
        row = row.astype('i', copy=copy)
        col = col.astype('i', copy=copy)

        if shape is None:
            if len(row) == 0 or len(col) == 0:
                raise ValueError(
                    'cannot infer dimensions from zero sized index arrays')
            shape = (int(row.max()) + 1, int(col.max()) + 1)

        if len(data) > 0:
            if row.max() >= shape[0]:
                raise ValueError('row index exceeds matrix dimensions')
            if col.max() >= shape[1]:
                raise ValueError('column index exceeds matrix dimensions')
            if row.min() < 0:
                raise ValueError('negative row index found')
            if col.min() < 0:
                raise ValueError('negative column index found')

        sparse_data._data_matrix.__init__(self, data)
        self.row = row
        self.col = col
        if not _util.isshape(shape):
            raise ValueError('invalid shape (must be a 2-tuple of int)')
        self._shape = int(shape[0]), int(shape[1])

    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return coo_matrix(
                (data, (self.row.copy(), self.col.copy())),
                shape=self.shape, dtype=data.dtype)
        else:
            return coo_matrix(
                (data, (self.row, self.col)), shape=self.shape,
                dtype=data.dtype)

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
            diag_coo = coo_matrix((self.data[diag_mask],
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

    def get_shape(self):
        """Returns the shape of the matrix.

        Returns:
            tuple: Shape of the matrix.
        """
        return self._shape

    def getnnz(self, axis=None):
        """Returns the number of stored values, including explicit zeros."""
        if axis is None:
            return self.data.size
        else:
            raise ValueError

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
        return scipy.sparse.coo_matrix(
            (data, (row, col)), shape=self.shape)

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

        new_data = self.data

        return coo_matrix((new_data, (new_row, new_col)), shape=shape,
                          copy=False)

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

        if diff[1:].all():
            # All elements have different indices.
            data = src_data
            row = src_row
            col = src_col
        else:
            # TODO(leofang): move the kernels outside this method
            index = cupy.cumsum(diff, dtype='i')
            size = int(index[-1]) + 1
            data = cupy.zeros(size, dtype=self.data.dtype)
            row = cupy.empty(size, dtype='i')
            col = cupy.empty(size, dtype='i')
            if self.data.dtype.kind == 'f':
                cupy.ElementwiseKernel(
                    'T src_data, int32 src_row, int32 src_col, int32 index',
                    'raw T data, raw int32 row, raw int32 col',
                    '''
                    atomicAdd(&data[index], src_data);
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'cupyx_scipy_sparse_coo_sum_duplicates_assign'
                )(src_data, src_row, src_col, index, data, row, col)
            elif self.data.dtype.kind == 'c':
                cupy.ElementwiseKernel(
                    'T src_real, T src_imag, int32 src_row, int32 src_col, '
                    'int32 index',
                    'raw T real, raw T imag, raw int32 row, raw int32 col',
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
        """Converts the matrix to COOdinate format.

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
        if self.nnz == 0:
            return _csc.csc_matrix(self.shape, dtype=self.dtype)
        # copy is silently ignored (in line with SciPy) because both
        # sum_duplicates and coosort change the underlying data
        x = self.copy()
        x.sum_duplicates()
        cusparse.coosort(x, 'c')
        x = cusparse.coo2csc(x)
        x.has_canonical_format = True
        return x

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in coo to csr conversion.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        if self.nnz == 0:
            return _csr.csr_matrix(self.shape, dtype=self.dtype)
        # copy is silently ignored (in line with SciPy) because both
        # sum_duplicates and coosort change the underlying data
        x = self.copy()
        x.sum_duplicates()
        cusparse.coosort(x, 'r')
        x = cusparse.coo2csr(x)
        x.has_canonical_format = True
        return x

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
        return coo_matrix(
            (self.data, (self.col, self.row)), shape=shape, copy=copy)


def isspmatrix_coo(x):
    """Checks if a given matrix is of COO format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.coo_matrix`.

    """
    return isinstance(x, coo_matrix)
