import numpy
try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

from cupy import cusparse
from cupy.sparse import base
from cupy.sparse import data as sparse_data


class coo_matrix(sparse_data._data_matrix):

    """COOrdinate format sparse matrix.

    Now it has only one initializer format below:

    ``coo_matrix((data, (row, col))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given data are always used.

    .. see::
       :class:`scipy.sparse.coo_matrix`

    """

    format = 'coo'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if shape is not None and len(shape) != 2:
            raise ValueError(
                'Only two-dimensional sparse arrays are supported.')

        if isinstance(arg1, tuple) and len(arg1) == 2:
            try:
                data, (row, col) = arg1
            except (TypeError, ValueError):
                raise TypeError('invalid input format')

            if not (base.isdense(data) and data.ndim == 1 and
                    base.isdense(row) and row.ndim == 1 and
                    base.isdense(col) and col.ndim == 1):
                raise ValueError('row, column, and data arrays must be 1-D')
            if not (len(data) == len(row) == len(col)):
                raise ValueError(
                    'row, column, and data array must all be the same length')

        elif isspmatrix_coo(arg1):
            data = arg1.data
            row = arg1.row
            col = arg1.col

            if shape is None:
                shape = arg1.shape

        else:
            raise ValueError(
                'Only (data, (row, col)) format is supported')

        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)

        if dtype != 'f' and dtype != 'd':
            raise ValueError('Only float32 and float64 are supported')

        data = data.astype(dtype, copy=copy)
        row = row.astype('i', copy=copy)
        col = col.astype('i', copy=copy)

        if shape is None:
            if len(row) == 0 or len(col) == 0:
                raise ValueError(
                    'cannot infer dimensions from zero sized index arrays')
            shape = (int(row.max()) + 1, int(col.max()) + 1)

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
        self._shape = shape

    def _with_data(self, data):
        return coo_matrix(
            (data, (self.row.copy(), self.col.copy())), shape=self.shape)

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

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order (str): Not supported.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same value.

        .. seealso:: :func:`cupy.sparse.coo_array.toarray`

        """
        return self.tocsr().toarray(order=order, out=out)

    def tocoo(self, copy=False):
        """Converts the matrix to COOdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupy.sparse.coo_matrix: Converte matrix.

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
            cupy.sparse.csc_matrix: Converte matrix.

        """
        return self.T.tocsr().T

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in coo to csr conversion.

        Returns:
            cupy.sparse.csr_matrix: Converte matrix.

        """
        # copy is ignored because coosort method breaks an original.
        x = self.copy()
        cusparse.coosort(x)
        return cusparse.coo2csr(x)

    def transpose(self, axes=None, copy=False):
        """Returns a transpose matrix.

        Args:
            axes: This option is not supported.
            copy (bool): If ``True``, a returned matrix shares no data.
                Otherwise, it shared data arrays as much as possible.

        """
        if axes is not None:
            raise ValueError(
                'Sparse matrices do not support an \'axes\' parameter because '
                'swapping dimensions is the only logical permutation.')
        shape = self.shape[1], self.shape[0]
        return coo_matrix(
            (self.data, (self.col, self.row)), shape=shape, copy=copy)


def isspmatrix_coo(x):
    """Checks if a given matrix is COO format.

    Returns:
        bool: Returns if ``x`` is :class:`cupy.sparse.coo_matrix`.

    """
    return isinstance(x, coo_matrix)
