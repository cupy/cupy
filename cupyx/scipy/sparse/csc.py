try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import cusparse
import cupyx.scipy.sparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import compressed


class csc_matrix(compressed._compressed_sparse_matrix):

    """Compressed Sparse Column matrix.

    Now it has only part of initializer formats:

    ``csc_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csc_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsc()``.
    ``csc_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csc_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
       :class:`scipy.sparse.csc_matrix`

    """

    format = 'csc'

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        .. warning::
           You need to install SciPy to use this method.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.csc_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        m = cusparse.dense2csc(x)
        return m.data, m.indices, m.indptr

    def _swap(self, x, y):
        return (y, x)

    # TODO(unno): Implement __getitem__

    def __mul__(self, other):
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif cupyx.scipy.sparse.isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm'):
                a = self.T
                return cusparse.csrgemm(a, other, transa=True)
            elif cusparse.check_availability('csrgemm2'):
                a = self.tocsr()
                a.sum_duplicates()
                return cusparse.csrgemm2(a, other)
            else:
                raise NotImplementedError
        elif isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm'):
                a = self.T
                b = other.T
                return cusparse.csrgemm(a, b, transa=True, transb=True)
            elif cusparse.check_availability('csrgemm2'):
                a = self.tocsr()
                b = other.tocsr()
                a.sum_duplicates()
                b.sum_duplicates()
                return cusparse.csrgemm2(a, b)
            else:
                raise NotImplementedError
        elif cupyx.scipy.sparse.isspmatrix(other):
            return self * other.tocsr()
        elif base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                if cusparse.check_availability('csrmv'):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv'):
                    csrmv = cusparse.spmv
                else:
                    raise NotImplementedError
                return csrmv(self.T, cupy.asfortranarray(other), transa=True)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2'):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise NotImplementedError
                return csrmm(self.T, cupy.asfortranarray(other), transa=True)
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    # TODO(unno): Implement check_format
    # TODO(unno): Implement diagonal

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        t = self.T
        t.eliminate_zeros()
        compress = t.T
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    # TODO(unno): Implement maximum
    # TODO(unno): Implement minimum
    # TODO(unno): Implement multiply
    # TODO(unno): Implement prune
    # TODO(unno): Implement reshape

    def sort_indices(self):
        """Sorts the indices of the matrix in place."""
        cusparse.cscsort(self)

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order ({'C', 'F', None}): Whether to store data in C (row-major)
                order or F (column-major) order. Default is C-order.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same matrix.

        .. seealso:: :meth:`scipy.sparse.csc_matrix.toarray`

        """
        if order is None:
            order = 'C'

        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)

        self.sum_duplicates()
        # csc2dense and csr2dense returns F-contiguous array.
        if order == 'C':
            # To return C-contiguous array, it uses transpose.
            return cusparse.csr2dense(self.T).T
        elif order == 'F':
            return cusparse.csc2dense(self)
        else:
            raise TypeError('order not understood')

    def _add_sparse(self, other, alpha, beta):
        self.sum_duplicates()
        other = other.tocsc().T
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self.T, other, alpha, beta).T

    # TODO(unno): Implement tobsr

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

        return cusparse.csc2coo(self, data, indices)

    def tocsc(self, copy=None):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, the method returns itself.
                Otherwise it makes a copy of the matrix.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in csr to csc conversion.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        # copy is ignored
        if cusparse.check_availability('csc2csr'):
            csc2csr = cusparse.csc2csr
        elif cusparse.check_availability('csc2csrEx2'):
            csc2csr = cusparse.csc2csrEx2
        else:
            raise NotImplementedError
        return csc2csr(self)

    # TODO(unno): Implement todia
    # TODO(unno): Implement todok
    # TODO(unno): Implement tolil

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
        trans = cupyx.scipy.sparse.csr.csr_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans._has_canonical_format = self._has_canonical_format
        return trans


def isspmatrix_csc(x):
    """Checks if a given matrix is of CSC format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csc_matrix`.

    """
    return isinstance(x, csc_matrix)
