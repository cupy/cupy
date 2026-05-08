from __future__ import annotations

try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import _core
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util


# TODO(leofang): The current implementation is CSC-based, which is troublesome
# on ROCm/HIP. We should convert it to CSR-based for portability.
class dia_matrix(_data._data_matrix):

    """Sparse matrix with DIAgonal storage.

    Now it has only one initializer format below:

    ``dia_matrix((data, offsets))``

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
       :class:`scipy.sparse.dia_matrix`

    """

    format = 'dia'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if _scipy_available and scipy.sparse.issparse(arg1):
            x = arg1.todia()
            data = x.data
            offsets = x.offsets
            shape = x.shape
            dtype = x.dtype
            copy = False
        elif isinstance(arg1, tuple):
            data, offsets = arg1
            if shape is None:
                raise ValueError('expected a shape argument')

        else:
            raise ValueError(
                'unrecognized form for dia_matrix constructor')

        data = cupy.array(data, dtype=dtype, copy=copy)
        data = cupy.atleast_2d(data)
        off_dtype = _sputils.get_index_dtype(maxval=max(shape))
        offsets = cupy.array(offsets, dtype=off_dtype)
        offsets = cupy.atleast_1d(offsets)

        if offsets.ndim != 1:
            raise ValueError('offsets array must have rank 1')

        if data.ndim != 2:
            raise ValueError('data array must have rank 2')

        if data.shape[0] != len(offsets):
            raise ValueError(
                'number of diagonals (%d) does not match the number of '
                'offsets (%d)'
                % (data.shape[0], len(offsets)))

        sorted_offsets = cupy.sort(offsets)
        if (sorted_offsets[:-1] == sorted_offsets[1:]).any():  # synchronize!
            raise ValueError('offset array contains duplicate values')

        self.data = data
        self.offsets = offsets
        if not _util.isshape(shape):
            raise ValueError('invalid shape (must be a 2-tuple of int)')
        self._shape = int(shape[0]), int(shape[1])

    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays are copied.
        """
        if copy:
            return dia_matrix((data, self.offsets.copy()), shape=self.shape)
        else:
            return dia_matrix((data, self.offsets), shape=self.shape)

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.dia_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        offsets = self.offsets.get(stream)
        return scipy.sparse.dia_matrix((data, offsets), shape=self._shape)

    def get_shape(self):
        """Returns the shape of the matrix.

        Returns:
            tuple: Shape of the matrix.
        """
        return self._shape

    def getnnz(self, axis=None):
        """Returns the number of stored values, including explicit zeros.

        Args:
            axis: Not supported yet.

        Returns:
            int: The number of stored values.

        """
        if axis is not None:
            raise NotImplementedError(
                'getnnz over an axis is not implemented for DIA format')

        m, n = self.shape
        # Bound by the actual data buffer length so an "empty" DIA
        # (data.shape[1] == 0 with non-empty offsets) reports 0,
        # matching scipy 1.17 (gh-23055).
        L = min(self.data.shape[1], n)
        it = self.offsets.dtype.type
        # Use int64 accumulator: per-diagonal counts fit int32, but the
        # sum across (m + n - 1) diagonals can exceed INT32_MAX even when
        # the offsets dtype is int32 (e.g., a dense 2**15 x 2**15 matrix).
        nnz = _core.ReductionKernel(
            'I offsets, I m, I L', 'int64 nnz',
            'max(min(m + offsets, L) - max(offsets, (I)0), (I)0)',
            'a + b', 'nnz = a', '0', 'dia_nnz')(
                self.offsets, it(m), it(L))
        return int(nnz)

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value."""
        return self.tocsc().toarray(order=order, out=out)

    def todia(self, copy=False):
        """Return this object unchanged (already in DIA format).

        The base ``_spbase.todia`` would round-trip via CSR, which is
        unnecessary work and currently raises ``NotImplementedError``
        because :meth:`csr_matrix.todia` is unimplemented.

        Args:
            copy (bool): If ``True``, return a copy.
        """
        if copy:
            return self.copy()
        return self

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in dia to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        if self.data.size == 0:
            return _csc.csc_matrix(self.shape, dtype=self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        idx_dtype = _sputils.get_index_dtype(maxval=max(self.shape))

        it = idx_dtype
        row, mask = _core.ElementwiseKernel(
            'I offset_len, I offsets, I num_rows, '
            'I num_cols, T data',
            'I row, bool mask',
            '''
            I offset_inds = (I)(i % offset_len);
            row = offset_inds - offsets;
            mask = (row >= 0 && row < num_rows
                    && offset_inds < num_cols
                    && data != T(0));
            ''',
            'cupyx_scipy_sparse_dia_tocsc')(
                it(offset_len),
                self.offsets[:, None].astype(idx_dtype, copy=False),
                it(num_rows), it(num_cols), self.data)
        indptr = cupy.zeros(num_cols + 1, dtype=idx_dtype)
        # When ``offset_len`` exceeds ``num_cols`` (data buffer wider
        # than the matrix), the trailing columns lie outside the matrix
        # and their mask entries are all False, so truncate to
        # ``num_cols`` for the indptr write.
        col_counts = mask.sum(axis=0)
        eff_len = min(offset_len, num_cols)
        indptr[1: eff_len + 1] = cupy.cumsum(col_counts[:eff_len])
        indptr[eff_len + 1:] = indptr[eff_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return _csc.csc_matrix(
            (data, indices, indptr), shape=self.shape, dtype=self.dtype)

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in dia to csr conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        return self.tocsc().tocsr()

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
        idx, = cupy.nonzero(self.offsets == k)
        first_col, last_col = max(0, k), min(rows + k, cols)
        if idx.size == 0:
            return cupy.zeros(last_col - first_col, dtype=self.data.dtype)
        return self.data[idx[0], first_col:last_col]


def isspmatrix_dia(x):
    """Checks if a given matrix is of DIA format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.dia_matrix`.

    """
    return isinstance(x, dia_matrix)
