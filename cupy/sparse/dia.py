try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import core
from cupy.sparse import data


class dia_matrix(data._data_matrix):

    """Sparse matrix with DIAgonal storage.

    Now it has only one initializer format below:

    ``dia_matrix((data, offsets))``

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. see::
       :class:`scipy.sparse.dia_matrix`

    """

    format = 'dia'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if isinstance(arg1, tuple):
            data, offsets = arg1
            if shape is None:
                raise ValueError('expected a shape argument')

        else:
            raise ValueError(
                'unrecognized form for dia_matrix constructor')

        data = cupy.array(data, dtype=dtype, copy=copy)
        data = cupy.atleast_2d(data)
        offsets = cupy.array(offsets, dtype='i', copy=copy)
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
        if (sorted_offsets[:-1] == sorted_offsets[1:]).any():
            raise ValueError('offset array contains duplicate values')

        self.data = data
        self.offsets = offsets
        self._shape = shape

    def _with_data(self, data):
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
        nnz = core.ReductionKernel(
            'int32 offsets, int32 m, int32 n', 'int32 nnz',
            'offsets > 0 ? min(m, n - offsets) : min(m + offsets, n)',
            'a + b', 'nnz = a', '0', 'dia_nnz')(self.offsets, m, n)
        return int(nnz)

def isspmatrix_dia(x):
    """Checks if a given matrix is of DIA format.

    Returns:
        bool: Returns if ``x`` is :class:`cupy.sparse.dia_matrix`.

    """
    return isinstance(x, dia_matrix)
