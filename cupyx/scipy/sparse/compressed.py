import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import core
from cupy.creation import basic
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import data as sparse_data
from cupyx.scipy.sparse import util


class _compressed_sparse_matrix(sparse_data._data_matrix):

    _compress_getitem_kern = core.ElementwiseKernel(
        'T d, S ind, int32 minor', 'raw T answer',
        'if (ind == minor) atomicAdd(&answer[0], d);',
        'compress_getitem')

    _compress_getitem_complex_kern = core.ElementwiseKernel(
        'T real, T imag, S ind, int32 minor',
        'raw T answer_real, raw T answer_imag',
        '''
        if (ind == minor) {
          atomicAdd(&answer_real[0], real);
          atomicAdd(&answer_imag[0], imag);
        }
        ''',
        'compress_getitem_complex')

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if shape is not None:
            if not util.isshape(shape):
                raise ValueError('invalid shape (must be a 2-tuple of int)')
            shape = int(shape[0]), int(shape[1])

        if base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            indices = x.indices
            indptr = x.indptr

            if arg1.format != self.format:
                # When formats are differnent, all arrays are already copied
                copy = False

            if shape is None:
                shape = arg1.shape

            has_canonical_format = x.has_canonical_format
        elif util.isshape(arg1):
            m, n = arg1
            m, n = int(m), int(n)
            data = basic.zeros(0, dtype if dtype else 'd')
            indices = basic.zeros(0, 'i')
            indptr = basic.zeros(self._swap(m, n)[0] + 1, dtype='i')
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False
            has_canonical_format = True

        elif scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse
            x = arg1.asformat(self.format)
            data = cupy.array(x.data)
            indices = cupy.array(x.indices, dtype='i')
            indptr = cupy.array(x.indptr, dtype='i')
            copy = False

            if shape is None:
                shape = arg1.shape
            has_canonical_format = x.has_canonical_format

        elif isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if not (base.isdense(data) and data.ndim == 1 and
                    base.isdense(indices) and indices.ndim == 1 and
                    base.isdense(indptr) and indptr.ndim == 1):
                raise ValueError(
                    'data, indices, and indptr should be 1-D')

            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')

            has_canonical_format = False

        elif base.isdense(arg1):
            if arg1.ndim > 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            elif arg1.ndim == 1:
                arg1 = arg1[None]
            elif arg1.ndim == 0:
                arg1 = arg1[None, None]
            data, indices, indptr = self._convert_dense(arg1)
            copy = False
            if shape is None:
                shape = arg1.shape

            has_canonical_format = True

        else:
            raise ValueError(
                'Unsupported initializer format')

        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)

        if dtype != 'f' and dtype != 'd' and dtype != 'F' and dtype != 'D':
            raise ValueError(
                'Only float32, float64, complex64 and complex128 '
                'are supported')

        data = data.astype(dtype, copy=copy)
        sparse_data._data_matrix.__init__(self, data)

        self.indices = indices.astype('i', copy=copy)
        self.indptr = indptr.astype('i', copy=copy)

        if shape is None:
            shape = self._swap(len(indptr) - 1, int(indices.max()) + 1)

        major, minor = self._swap(*shape)
        if len(indptr) != major + 1:
            raise ValueError('index pointer size (%d) should be (%d)'
                             % (len(indptr), major + 1))

        self._descr = cusparse.MatDescriptor.create()
        self._shape = shape
        self._has_canonical_format = has_canonical_format

    def _with_data(self, data, copy=True):
        if copy:
            return self.__class__(
                (data, self.indices.copy(), self.indptr.copy()),
                shape=self.shape,
                dtype=data.dtype)
        else:
            return self.__class__(
                (data, self.indices, self.indptr),
                shape=self.shape,
                dtype=data.dtype)

    def _convert_dense(self, x):
        raise NotImplementedError

    def _swap(self, x, y):
        raise NotImplementedError

    def _add_sparse(self, other, alpha, beta):
        raise NotImplementedError

    def _add(self, other, lhs_negative, rhs_negative):
        if cupy.isscalar(other):
            if other == 0:
                if lhs_negative:
                    return -self
                else:
                    return self.copy()
            else:
                raise NotImplementedError(
                    'adding a nonzero scalar to a sparse matrix is not '
                    'supported')
        elif base.isspmatrix(other):
            alpha = -1 if lhs_negative else 1
            beta = -1 if rhs_negative else 1
            return self._add_sparse(other, alpha, beta)
        elif base.isdense(other):
            if lhs_negative:
                if rhs_negative:
                    return -self.todense() - other
                else:
                    return other - self.todense()
            else:
                if rhs_negative:
                    return self.todense() - other
                else:
                    return self.todense() + other
        else:
            return NotImplemented

    def __add__(self, other):
        return self._add(other, False, False)

    def __radd__(self, other):
        return self._add(other, False, False)

    def __sub__(self, other):
        return self._add(other, False, True)

    def __rsub__(self, other):
        return self._add(other, True, False)

    def __getitem__(self, slices):
        if isinstance(slices, tuple):
            slices = list(slices)
        elif isinstance(slices, list):
            slices = list(slices)
            if all([isinstance(s, int) for s in slices]):
                slices = [slices]
        else:
            slices = [slices]

        ellipsis = -1
        n_ellipsis = 0
        for i, s in enumerate(slices):
            if s is None:
                raise IndexError('newaxis is not supported')
            elif s is Ellipsis:
                ellipsis = i
                n_ellipsis += 1
        if n_ellipsis > 0:
            ellipsis_size = self.ndim - (len(slices) - 1)
            slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size

        if len(slices) == 2:
            row, col = slices
        elif len(slices) == 1:
            row, col = slices[0], slice(None)
        else:
            raise IndexError('invalid number of indices')

        major, minor = self._swap(row, col)
        major_size, minor_size = self._swap(*self._shape)
        if numpy.isscalar(major):
            i = int(major)
            if i < 0:
                i += major_size
            if not (0 <= i < major_size):
                raise IndexError('index out of bounds')
            if numpy.isscalar(minor):
                j = int(minor)
                if j < 0:
                    j += minor_size
                if not (0 <= j < minor_size):
                    raise IndexError('index out of bounds')
                return self._get_single(i, j)
            elif minor == slice(None):
                return self._get_major_slice(slice(i, i + 1))
        elif isinstance(major, slice):
            if minor == slice(None):
                return self._get_major_slice(major)

        raise ValueError('unsupported indexing')

    def _get_single(self, major, minor):
        start = self.indptr[major]
        end = self.indptr[major + 1]
        answer = cupy.zeros((), self.dtype)
        data = self.data[start:end]
        indices = self.indices[start:end]
        if self.dtype.kind == 'c':
            self._compress_getitem_complex_kern(
                data.real, data.imag, indices, minor, answer.real, answer.imag)
        else:
            self._compress_getitem_kern(
                data, indices, minor, answer)
        return answer[()]

    def _get_major_slice(self, major):
        major_size, minor_size = self._swap(*self._shape)
        # major.indices cannot be used because scipy.sparse behaves differently
        major_start = major.start
        major_stop = major.stop
        major_step = major.step
        if major_start is None:
            major_start = 0
        if major_stop is None:
            major_stop = major_size
        if major_step is None:
            major_step = 1
        if major_start < 0:
            major_start += major_size
        if major_stop < 0:
            major_stop += major_size
        major_start = max(min(major_start, major_size), 0)
        major_stop = max(min(major_stop, major_size), 0)

        if major_step != 1:
            raise ValueError('slicing with step != 1 not supported')

        if not (major_start <= major_stop):
            raise IndexError('index out of bounds')

        start = self.indptr[major_start]
        stop = self.indptr[major_stop]
        data = self.data[start:stop]
        indptr = self.indptr[major_start:major_stop + 1] - start
        indices = self.indices[start:stop]

        shape = self._swap(len(indptr) - 1, minor_size)
        return self.__class__(
            (data, indices, indptr), shape=shape, dtype=self.dtype, copy=False)

    @property
    def has_canonical_format(self):
        return self._has_canonical_format

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

    def sum_duplicates(self):
        if self._has_canonical_format:
            return
        if self.data.size == 0:
            self._has_canonical_format = True
            return
        coo = self.tocoo()
        coo.sum_duplicates()
        self.__init__(coo.asformat(self.format))
        self._has_canonical_format = True
