import numpy

from cupy import cusparse
from cupy.sparse import base
from cupy.sparse import data as sparse_data


class _compressed_sparse_matrix(sparse_data._data_matrix):

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if shape is not None and len(shape) != 2:
                raise ValueError(
                    'Only two-dimensional sparse arrays are supported.')

            if not (base.isdense(data) and data.ndim == 1 and
                    base.isdense(indices) and indices.ndim == 1 and
                    base.isdense(indptr) and indptr.ndim == 1):
                raise ValueError(
                    'data, indices, and indptr should be 1-D')

            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')

            if dtype is None:
                dtype = data.dtype
            else:
                dtype = numpy.dtype(dtype)

            if dtype != 'f' and dtype != 'd':
                raise ValueError('Only float32 and float64 are supported')

            data = data.astype(dtype, copy=copy)
            sparse_data._data_matrix.__init__(self, data)

            self.indices = indices.astype('i', copy=copy)
            self.indptr = indptr.astype('i', copy=copy)

            if shape is None:
                shape = self._swap(len(indptr) - 1, int(indices.max()) + 1)
        else:
            raise ValueError(
                'Only (data, indices, indptr) format is supported')

        major, minor = self._swap(*shape)
        if len(indptr) != major + 1:
            raise ValueError('index pointer size (%d) should be (%d)'
                             % (len(indptr), major + 1))

        self._descr = cusparse.MatDescriptor.create()
        self._shape = shape

    def _with_data(self, data):
        return self.__class__(
            (data, self.indices.copy(), self.indptr.copy()), shape=self.shape)

    def _swap(self, x, y):
        raise NotImplementedError

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
        if axis is None:
            return self.data.size
        else:
            raise ValueError

    # TODO(unno): Implement sorted_indices
