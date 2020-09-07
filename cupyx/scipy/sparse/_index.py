"""Indexing mixin for sparse matrix classes.
"""

import cupy
from cupy import core

from cupyx.scipy.sparse.base import isspmatrix
from cupyx.scipy.sparse.base import spmatrix

from cupy_backends.cuda.libs import cusparse
from cupy.cuda import device

import numpy

try:
    import scipy
    scipy_available = True
except ImportError:
    scipy_available = False

_int_scalar_types = (int, numpy.integer, numpy.int_)
_bool_scalar_types = (bool, numpy.bool, numpy.bool_)


def _get_csr_submatrix_major_axis(Ap, Aj, Ax, start, stop):
    """Return a submatrix of the input sparse matrix by slicing major axis.

    Args:
        Ap (cupy.ndarray): indptr array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ax (cupy.ndarray): data array from input sparse matrix
        start (int): starting index of major axis
        stop (int): ending index of major axis

    Returns:
        Bp (cupy.ndarray): indptr array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bx (cupy.ndarray): data array of output sparse matrix

    """
    Ap = Ap[start:stop + 1]
    start_offset, stop_offset = int(Ap[0]), int(Ap[-1])
    Bp = Ap - start_offset
    Bj = Aj[start_offset:stop_offset]
    Bx = Ax[start_offset:stop_offset]

    return Bp, Bj, Bx


def _get_csr_submatrix_minor_axis(Ap, Aj, Ax, start, stop):
    """Return a submatrix of the input sparse matrix by slicing minor axis.

    Args:
        Ap (cupy.ndarray): indptr array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ax (cupy.ndarray): data array from input sparse matrix
        start (int): starting index of minor axis
        stop (int): ending index of minor axis

    Returns:
        Bp (cupy.ndarray): indptr array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bx (cupy.ndarray): data array of output sparse matrix

    """
    mask = (start <= Aj) & (Aj < stop)
    mask_sum = cupy.empty(Aj.size + 1, dtype=Aj.dtype)
    mask_sum[0] = 0
    mask_sum[1:] = mask
    cupy.cumsum(mask_sum, out=mask_sum)
    Bp = mask_sum[Ap]
    Bj = Aj[mask] - start
    Bx = Ax[mask]

    return Bp, Bj, Bx


_csr_row_index_ker = core.ElementwiseKernel(
    'int32 out_rows, raw I rows, '
    'raw int32 Ap, raw int32 Aj, raw T Ax, raw int32 Bp',
    'int32 Bj, T Bx',
    '''
    const I row = rows[out_rows];

    // Look up starting offset
    const I starting_output_offset = Bp[out_rows];
    const I output_offset = i - starting_output_offset;
    const I starting_input_offset = Ap[row];

    Bj = Aj[starting_input_offset + output_offset];
    Bx = Ax[starting_input_offset + output_offset];
''', 'csr_row_index_ker')


def _csr_row_index(rows, Ap, Aj, Ax):
    """Populate indices and data arrays from the given row index

    Args:
        rows (cupy.ndarray): index array of rows to populate
        Ap (cupy.ndarray): indptr array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ax (cupy.ndarray): data array from input sparse matrix

    Returns:
        Bp (cupy.ndarray): indptr array for output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bx (cupy.ndarray): data array of output sparse matrix

    """
    row_nnz = cupy.diff(Ap)
    Bp = cupy.empty(rows.size + 1, dtype=Ap.dtype)
    Bp[0] = 0
    cupy.cumsum(row_nnz[rows], out=Bp[1:])
    nnz = int(Bp[-1])

    out_rows = cupy.empty(nnz, dtype=numpy.int32)

    # Build a COO row array from output CSR indptr.
    # Calling backend cusparse API directly to avoid
    # constructing a whole COO object.
    handle = device.get_cusparse_handle()
    cusparse.xcsr2coo(
        handle, Bp.data.ptr, nnz, Bp.size-1, out_rows.data.ptr,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)

    Bj, Bx = _csr_row_index_ker(out_rows, rows, Ap, Aj, Ax, Bp)
    return Bp, Bj, Bx


def _csr_sample_values(n_row, n_col,
                       Ap, Aj, Ax,
                       Bi, Bj):
    """Populate data array for a set of rows and columns
    Args
        n_row : total number of rows in input array
        n_col : total number of columns in input array
        Ap : indptr array for input sparse matrix
        Aj : indices array for input sparse matrix
        Ax : data array for input sparse matrix
        Bi : array of rows to extract from input sparse matrix
        Bj : array of columns to extract from input sparse matrix
    Returns
        Bx : data array for output sparse matrix
    """

    Bi[Bi < 0] += n_row
    Bj[Bj < 0] += n_col

    Bx = cupy.empty(Bi.size, dtype=Ax.dtype)
    _csr_sample_values_kern(n_row, n_col,
                            Ap, Aj, Ax,
                            Bi, Bj, Bx, size=Bi.size)

    return Bx


_csr_sample_values_kern = core.ElementwiseKernel(
    '''I n_row, I n_col, raw I Ap, raw I Aj, raw T Ax, raw I Bi, raw I Bj''',
    'raw T Bx', '''
    const I j = Bi[i]; // sample row
    const I k = Bj[i]; // sample column
    const I row_start = Ap[j];
    const I row_end   = Ap[j+1];
    T x = 0;
    for(I jj = row_start; jj < row_end; jj++) {
        if (Aj[jj] == k)
            x += Ax[jj];
    }
    Bx[i] = x;
''', 'csr_sample_values_kern', no_return=True)


class IndexMixin(object):
    """
    This class provides common dispatching and validation logic for indexing.
    """

    def __getitem__(self, key):

        # For testing- Scipy >= 1.4.0 is needed to guarantee
        # results match.
        if scipy_available and numpy.lib.NumpyVersion(
                scipy.__version__) < '1.4.0':
            raise NotImplementedError(
                "Sparse __getitem__() requires Scipy >= 1.4.0")

        row, col = self._parse_indices(key)

        # Dispatch to specialized methods.
        if isinstance(row, _int_scalar_types):
            if isinstance(col, _int_scalar_types):
                return self._get_intXint(row, col)
            elif isinstance(col, slice):
                return self._get_intXslice(row, col)
            elif col.ndim == 1:
                return self._get_intXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif isinstance(row, slice):
            if isinstance(col, _int_scalar_types):
                return self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    return self.copy()
                return self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                return self._get_sliceXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif row.ndim == 1:
            if isinstance(col, _int_scalar_types):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                return self._get_arrayXslice(row, col)
        else:  # row.ndim == 2
            if isinstance(col, _int_scalar_types):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                raise IndexError('index results in >2 dimensions')
            elif row.shape[1] == 1 and (col.ndim == 1 or col.shape[0] == 1):
                # special case for outer indexing
                return self._get_columnXarray(row[:, 0], col.ravel())

        # The only remaining case is inner (fancy) indexing
        row, col = cupy.broadcast_arrays(row, col)
        if row.shape != col.shape:
            raise IndexError('number of row and column indices differ')
        if row.size == 0:
            return self.__class__(cupy.atleast_2d(row).shape, dtype=self.dtype)
        return self._get_arrayXarray(row, col)

    def _is_scalar(self, index):
        if isinstance(index, (cupy.ndarray, numpy.ndarray)) and \
                index.ndim == 0 and index.size == 1:
            return True
        return False

    def _parse_indices(self, key):
        M, N = self.shape
        row, col = _unpack_index(key)

        # Scipy calls sputils.isintlike() rather than
        # isinstance(x, _int_scalar_types). Comparing directly to int
        # here to minimize the impact of nested exception catching

        if self._is_scalar(row):
            row = row.item()
        if self._is_scalar(col):
            col = col.item()

        if isinstance(row, _int_scalar_types):
            row = _normalize_index(row, M, 'row')
        elif not isinstance(row, slice):
            row = self._asindices(row, M)

        if isinstance(col, _int_scalar_types):
            col = _normalize_index(col, N, 'column')
        elif not isinstance(col, slice):
            col = self._asindices(col, N)

        return row, col

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.

        idx is assumed to be at least a 1-dimensional array-like, but can
        have no more than 2 dimensions.
        """
        try:
            x = cupy.asarray(idx, dtype=self.indices.dtype)
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')

        return x % length

    def getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single row
        """
        M, N = self.shape
        i = _normalize_index(i, M, 'index')
        return self._get_intXslice(i, slice(None))

    def getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single column
        """
        M, N = self.shape
        i = _normalize_index(i, N, 'index')
        return self._get_sliceXint(slice(None), i)

    def _get_intXint(self, row, col):
        raise NotImplementedError()

    def _get_intXarray(self, row, col):
        raise NotImplementedError()

    def _get_intXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXint(self, row, col):
        raise NotImplementedError()

    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXint(self, row, col):
        raise NotImplementedError()

    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()

    def _get_columnXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()

    def _set_intXint(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray_sparse(self, row, col, x):
        # Fall back to densifying x
        x = cupy.asarray(x.toarray(), dtype=self.dtype)
        x, _ = cupy.broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)


def _try_is_scipy_spmatrix(index):
    if scipy_available:
        return isinstance(index, scipy.sparse.base.spmatrix)
    return False


def _unpack_index(index):
    """ Parse index. Always return a tuple of the form (row, col).
    Valid type for row/col is integer, slice, or array of integers.

    Returns:
          resulting row & col indices : single integer, slice, or
          array of integers. If row & column indices are supplied
          explicitly, they are used as the major/minor indices.
          If only one index is supplied, the minor index is
          assumed to be all (e.g., [maj, :]).
    """
    # First, check if indexing with single boolean matrix.
    if ((isinstance(index, (spmatrix, cupy.ndarray,
                            numpy.ndarray))
         or _try_is_scipy_spmatrix(index))
            and index.ndim == 2 and index.dtype.kind == 'b'):
        return index.nonzero()

    # Parse any ellipses.
    index = _eliminate_ellipsis(index)

    # Next, parse the tuple or object
    if isinstance(index, tuple):
        if len(index) == 2:
            row, col = index
        elif len(index) == 1:
            row, col = index[0], slice(None)
        else:
            raise IndexError('invalid number of indices')
    else:
        idx = _compatible_boolean_index(index)
        if idx is None:
            row, col = index, slice(None)
        elif idx.ndim < 2:
            return _boolean_index_to_array(idx), slice(None)
        elif idx.ndim == 2:
            return idx.nonzero()
    # Next, check for validity and transform the index as needed.
    if isspmatrix(row) or isspmatrix(col):
        # Supporting sparse boolean indexing with both row and col does
        # not work because spmatrix.ndim is always 2.
        raise IndexError(
            'Indexing with sparse matrices is not supported '
            'except boolean indexing where matrix and index '
            'are equal shapes.')
    bool_row = _compatible_boolean_index(row)
    bool_col = _compatible_boolean_index(col)
    if bool_row is not None:
        row = _boolean_index_to_array(bool_row)
    if bool_col is not None:
        col = _boolean_index_to_array(bool_col)
    return row, col


def _eliminate_ellipsis(index):
    """Process indices with Ellipsis. Returns modified index."""
    if index is Ellipsis:
        return (slice(None), slice(None))

    if not isinstance(index, tuple):
        return index

    # Find first ellipsis.
    for j, v in enumerate(index):
        if v is Ellipsis:
            first_ellipsis = j
            break
    else:
        return index

    # Try to expand it using shortcuts for common cases
    if len(index) == 1:
        return (slice(None), slice(None))
    if len(index) == 2:
        if first_ellipsis == 0:
            if index[1] is Ellipsis:
                return (slice(None), slice(None))
            return (slice(None), index[1])
        return (index[0], slice(None))

    # Expand it using a general-purpose algorithm
    tail = []
    for v in index[first_ellipsis+1:]:
        if v is not Ellipsis:
            tail.append(v)
    nd = first_ellipsis + len(tail)
    nslice = max(0, 2 - nd)
    return index[:first_ellipsis] + (slice(None),) * nslice + tuple(tail,)


def _normalize_index(x, dim, name):
    if x < -dim or x >= dim:
        raise IndexError('{} ({}) out of range'.format(name, x))
    if x < 0:
        x += dim
    return x


def _first_element_bool(idx, max_dim=2):
    """Returns True if first element of the incompatible
    array type is boolean.
    """
    if max_dim < 1:
        return None
    try:
        first = idx[0] if len(idx) > 0 else None
    except TypeError:
        return None
    if isinstance(first, _bool_scalar_types):
        return True
    return _first_element_bool(first, max_dim-1)


def _compatible_boolean_index(idx):
    """Returns a boolean index array that can be converted to
    integer array. Returns None if no such array exists.
    """
    # presence of attribute `ndim` indicates a compatible array type.
    if hasattr(idx, 'ndim'):
        if idx.dtype.kind == 'b':
            return idx
    # non-ndarray bool collection should be converted to ndarray
    elif _first_element_bool(idx):
        return cupy.asarray(idx, dtype='bool')
    return None


def _boolean_index_to_array(idx):
    if idx.ndim > 1:
        raise IndexError('invalid index shape')
    idx = cupy.array(idx, dtype=idx.dtype)
    return cupy.where(idx)[0]
