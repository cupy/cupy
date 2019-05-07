try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import compressed
from cupyx.scipy.sparse import csc


class csr_matrix(compressed._compressed_sparse_matrix):

    """Compressed Sparse Row matrix.

    Now it has only part of initializer formats:

    ``csr_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csr_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsr()``.
    ``csr_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
       :class:`scipy.sparse.csr_matrix`

    """

    format = 'csr'

    # TODO(unno): Implement has_sorted_indices

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.csr_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        m = cusparse.dense2csr(x)
        return m.data, m.indices, m.indptr

    def _swap(self, x, y):
        return (x, y)

    # TODO(unno): Implement __getitem__

    def _add_sparse(self, other, alpha, beta):
        self.sum_duplicates()
        other = other.tocsr()
        other.sum_duplicates()
        return cusparse.csrgeam(self, other, alpha, beta)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return cusparse.csrgemm(self, other)
        elif csc.isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return cusparse.csrgemm(self, other.T, transb=True)
        elif base.isspmatrix(other):
            return self * other.tocsr()
        elif base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                return cusparse.csrmv(self, cupy.asfortranarray(other))
            elif other.ndim == 2:
                self.sum_duplicates()
                return cusparse.csrmm2(self, cupy.asfortranarray(other))
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    # TODO(unno): Implement argmax
    # TODO(unno): Implement argmin
    # TODO(unno): Implement check_format

    def diagonal(self):
        # TODO(unno): Implement diagonal
        raise NotImplementedError

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        compress = cusparse.csr2csr_compress(self, 0)
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    # TODO(unno): Implement max

    def maximum(self, other):
        # TODO(unno): Implement maximum
        raise NotImplementedError

    # TODO(unno): Implement min

    def minimum(self, other):
        # TODO(unno): Implement minimum
        raise NotImplementedError

    def multiply(self, other):
        # TODO(unno): Implement multiply
        raise NotImplementedError

    # TODO(unno): Implement prune
    # TODO(unno): Implement reshape

    def sort_indices(self):
        """Sorts the indices of the matrix in place."""
        cusparse.csrsort(self)

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order ({'C', 'F', None}): Whether to store data in C (row-major)
                order or F (column-major) order. Default is C-order.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same matrix.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.toarray`

        """
        if order is None:
            order = 'C'

        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)

        self.sum_duplicates()
        # csr2dense returns F-contiguous array.
        if order == 'C':
            # To return C-contiguous array, it uses transpose.
            return cusparse.csc2dense(self.T).T
        elif order == 'F':
            return cusparse.csr2dense(self)
        else:
            raise TypeError('order not understood')

    def tobsr(self, blocksize=None, copy=False):
        # TODO(unno): Implement tobsr
        raise NotImplementedError

    def tocoo(self, copy=False):
        """Converts the matrix to COOdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupyx.scipy.sparse.coo_matrix: Converted matrix.

        """
        if copy:
            data = self.data.copy()
            indices = self.indices.copy()
        else:
            data = self.data
            indices = self.indices

        return cusparse.csr2coo(self, data, indices)

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in csr to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        # copy is ignored
        return cusparse.csr2csc(self)

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, the method returns itself.
                Otherwise it makes a copy of the matrix.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    def todia(self, copy=False):
        # TODO(unno): Implement todia
        raise NotImplementedError

    def todok(self, copy=False):
        # TODO(unno): Implement todok
        raise NotImplementedError

    def tolil(self, copy=False):
        # TODO(unno): Implement tolil
        raise NotImplementedError

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
        return csc.csc_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)


def isspmatrix_csr(x):
    """Checks if a given matrix is of CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)
