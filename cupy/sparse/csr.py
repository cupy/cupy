try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

from cupy import cusparse
from cupy.sparse import compressed
from cupy.sparse import csc


class csr_matrix(compressed._compressed_sparse_matrix):

    """Compressed Sparse Row matrix.

    Now it has only one initializer format below:

    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool):

    .. see::
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

    def _swap(self, x, y):
        return (x, y)

    # TODO(unno): Implement __getitem__

    # TODO(unno): Implement argmax
    # TODO(unno): Implement argmin
    # TODO(unno): Implement check_format
    # TODO(unno): Implement diagonal
    # TODO(unno): Implement dot
    # TODO(unno): Implement eliminate_zeros

    # TODO(unno): Implement max
    # TODO(unno): Implement maximum
    # TODO(unno): Implement min
    # TODO(unno): Implement minimum
    # TODO(unno): Implement multiply
    # TODO(unno): Implement prune
    # TODO(unno): Implement reshape

    def sort_indices(self):
        """Sort the indices of the matrix in place."""
        cusparse.csrsort(self)

    # TODO(unno): Implement sum_duplicates

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order (str): Not supported.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same value.

        .. seealso:: :func:`cupy.sparse.csr_array.toarray`

        """
        return cusparse.csr2dense(self)

    # TODO(unno): Implement tobsr
    # TODO(unno): Implement tocoo

    def tocsc(self, copy=False):
        """Convert the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupy.sparse.csc_matrix: Converte matrix.

        """
        # copy is ignored
        return cusparse.csr2csc(self)

    def tocsr(self, copy=False):
        """Convert the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupy.sparse.csr_matrix: Converte matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    # TODO(unno): Implement todia
    # TODO(unno): Implement todok
    # TODO(unno): Implement tolil

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
        return csc.csc_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)


def isspmatrix_csr(x):
    """Check if a given matrix is CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)
