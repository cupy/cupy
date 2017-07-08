try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy import cusparse
from cupy.sparse import compressed


class csc_matrix(compressed._compressed_sparse_matrix):

    """Compressed Sparse Column matrix.

    Now it has only one initializer format below:

    ``csc_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. see::
       :class:`scipy.sparse.csc_matrix`

    """

    format = 'csc'

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        .. warn::
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

    def _swap(self, x, y):
        return (y, x)

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
        """Sorts the indices of the matrix in place."""
        cusparse.cscsort(self)

    # TODO(unno): Implement sum_duplicates

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order: Not supported.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same matrix.

        .. seealso:: :func:`cupy.sparse.csc_array.toarray`

        """
        return self.T.toarray().T

    # TODO(unno): Implement tobsr

    def tocoo(self, copy=False):
        """Converts the matrix to COOdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupy.sparse.coo_matrix: Converted matrix.

        """
        return self.T.tocoo(copy).T

    def tocsc(self, copy=None):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy: Not supported yet.

        Returns:
            cupy.sparse.csc_matrix: Converted matrix.

        """
        return self

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in csr to csc conversion.

        Returns:
            cupy.sparse.csr_matrix: Converted matrix.

        """
        return self.T.tocsc(copy=False).T

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
            cupy.sparse.spmatrix: Transpose matrix.

        """
        if axes is not None:
            raise ValueError(
                'Sparse matrices do not support an \'axes\' parameter because '
                'swapping dimensions is the only logical permutation.')

        shape = self.shape[1], self.shape[0]
        return cupy.sparse.csr_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)


def isspmatrix_csc(x):
    """Checks if a given matrix is of CSC format.

    Returns:
        bool: Returns if ``x`` is :class:`cupy.sparse.csc_matrix`.

    """
    return isinstance(x, csc_matrix)
