"""Indexing mixin for sparse matrix classes.
"""
from .sputils import isintlike

from numpy import integer

import cupy
from cupy import core


INT_TYPES = (int, integer)


def _csr_column_inv_idx(idxs):
    """Construct an inverted index, mapping the indices
    of the given array to the their values

    Args
        idxs : array of indices to invert
        tpb : threads per block for underlying kernel

    Returns
        idxs_adj : inverted indices where idxs_adj[idxs[i]] = i
    """
    max_idx = idxs.max().item()
    idxs_adj = cupy.zeros(max_idx + 1, dtype=idxs.dtype)
    idxs_adj[idxs] = cupy.arange(idxs.size)

    return idxs_adj


def _get_csr_submatrix(Ap, Aj, Ax,
                       start_maj, stop_maj,
                       start_min, stop_min):
    """Return a submatrix of the input sparse matrix by
    slicing both major and minor axes.

    Args
        Ap : indptr array from input sparse matrix
        Aj : indices array from input sparse matrix
        Ax : data array from input sparse matrix
        start_maj : starting index of major axis
        stop_maj : ending index of major axis
        start_min : starting index of minor axis
        stop_min : ending index of minor axis

    Returns
        Bp : indptr array of output sparse matrix
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """

    Bp = Ap[start_maj:stop_maj+1]

    Aj = Aj[Bp[0]:Bp[-1]]
    Ax = Ax[Bp[0]:Bp[-1]]

    Aj_copy = cupy.zeros(Aj.size+1, dtype=Aj.dtype)
    Aj_copy[:-1] = Aj

    Aj_copy[cupy.where((Aj < start_min) | (Aj >= stop_min))] = -1

    mask = Aj_copy > -1

    Aj_copy[mask] = 1
    Aj_copy[~mask] = 0
    Aj_copy[1:] = Aj_copy[:-1]
    Aj_copy[0] = 0

    cupy.cumsum(Aj_copy, out=Aj_copy)
    Bp = Aj_copy[Bp-Bp[0]]

    Bj = Aj[mask[:-1]] - start_min
    Bx = Ax[mask[:-1]]

    return Bp, Bj, Bx


_csr_row_index_ker = core.ElementwiseKernel(
    '''raw I rows, raw I Ap, raw I Aj, raw T Ax, raw I Bp''',
    '''raw I Bj, raw T Bx''', '''
    const I row = rows[i];
    const I row_start = Ap[row];
    const I row_end = Ap[row+1];

    I out_row_idx = Bp[i];

    // Copy columns
    for(I j = row_start; j < row_end; j++) {
        Bj[out_row_idx] = Aj[j];
        Bx[out_row_idx] = Ax[j];
        out_row_idx++;
    }
''', 'csr_row_index_ker', no_return=True)


def _csr_row_index(rows,
                   Ap, Aj, Ax,
                   Bp):
    """Populate indices and data arrays from the given row index

    Args
        rows : index array of rows to populate
        Ap : indptr array from input sparse matrix
        Aj : indices array from input sparse matrix
        Ax : data array from input sparse matrix
        Bp : indptr array for output sparse matrix
        tpb : threads per block of row index kernel

    Returns
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """

    nnz = Bp[-1].item()
    Bj = cupy.empty(nnz, dtype=Aj.dtype)
    Bx = cupy.empty(nnz, dtype=Ax.dtype)

    _csr_row_index_ker(rows, Ap, Aj, Ax, Bp, Bj, Bx, size=rows.size)

    return Bj, Bx


_set_boolean_mask_for_offsets = core.ElementwiseKernel(
    'raw T start_offsets, raw T stop_offsets', 'raw bool mask',
    '''
    for (int jj = start_offsets[i]; jj < stop_offsets[i]; jj++) {
        mask[jj] = true;
    }
    ''', 'set_boolean_mask_for_offsets', no_return=True)


def _csr_row_slice(start, step, Ap, Aj, Ax, Bp):
    """Populate indices and data arrays of sparse matrix by slicing the
    rows of an input sparse matrix

    Args
        start : starting row
        step : step increment size
        Ap : indptr array of input sparse matrix
        Aj : indices array of input sparse matrix
        Ax : data array of input sparse matrix
        Bp : indices array of output sparse matrix

    Returns
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """

    in_rows = cupy.arange(Bp.size-1, dtype=Bp.dtype) * step + start
    start_offsets = Ap[in_rows]
    stop_offsets = Ap[in_rows+1]

    Aj_mask = cupy.zeros_like(Aj, dtype='bool')

    _set_boolean_mask_for_offsets(
        start_offsets, stop_offsets, Aj_mask, size=start_offsets.size)

    Bj = Aj[Aj_mask]
    Bx = Ax[Aj_mask]

    if step < 0:
        Bj = Bj[::-1].copy()
        Bx = Bx[::-1].copy()

    return Bj, Bx


class IndexMixin(object):
    """
    This class provides common dispatching and validation logic for indexing.
    """

    def __getitem__(self, key):
        row, col = self._validate_indices(key)
        # Dispatch to specialized methods.
        if isinstance(row, INT_TYPES):
            if isinstance(col, INT_TYPES):
                return self._get_intXint(row, col)
            elif isinstance(col, slice):
                return self._get_intXslice(row, col)
            elif col.ndim == 1:
                return self._get_intXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif isinstance(row, slice):
            if isinstance(col, INT_TYPES):
                return self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    return self.copy()
                return self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                return self._get_sliceXarray(row, col)
            raise IndexError('index results in >2 dimensions')
        elif row.ndim == 1:
            if isinstance(col, INT_TYPES):
                return self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                return self._get_arrayXslice(row, col)
        else:  # row.ndim == 2
            if isinstance(col, INT_TYPES):
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

    def _validate_indices(self, key):
        M, N = self.shape
        row, col = _unpack_index(key)

        if isintlike(row):
            row = int(row)
            if row < -M or row >= M:
                raise IndexError('row index (%d) out of range' % row)
            if row < 0:
                row += M
        elif not isinstance(row, slice):
            row = self._asindices(row, M)

        if isintlike(col):
            col = int(col)
            if col < -N or col >= N:
                raise IndexError('column index (%d) out of range' % col)
            if col < 0:
                col += N
        elif not isinstance(col, slice):
            col = self._asindices(col, N)

        return row, col

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.
        """
        try:
            x = cupy.asarray(idx, dtype="int32")
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')

        if x.size == 0:
            return x

        if x.max().item() > length:
            raise IndexError('index (%d) out of range' % x.max())

        min_item = x.min().item()
        if min_item < 0:
            if min_item < -length:
                raise IndexError('index (%d) out of range' % x.min())
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    def getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single row
        """
        M, N = self.shape
        i = int(i)
        if i < -M or i >= M:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += M
        return self._get_intXslice(i, slice(None))

    def getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.spmatrix: Sparse matrix with single column
        """
        M, N = self.shape
        i = int(i)
        if i < -N or i >= N:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += N
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


def _unpack_index(index):
    """ Parse index. Always return a tuple of the form (row, col).
    Valid type for row/col is integer, slice, or array of integers.
    """
    # First, check if indexing with single boolean matrix.
    from .base import spmatrix, isspmatrix
    if (isinstance(index, (spmatrix, cupy.ndarray)) and
            index.ndim == 2 and index.dtype.kind == 'b'):
        return index.nonzero()

    # Parse any ellipses.
    index = _check_ellipsis(index)

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


def _check_ellipsis(index):
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
    return index[:first_ellipsis] + (slice(None),)*nslice + tuple(tail)


def _maybe_bool_ndarray(idx):
    """Returns a compatible array if elements are boolean.
    """
    idx = cupy.asanyarray(idx)
    if idx.dtype.kind == 'b':
        return idx
    return None


def _first_element_bool(idx, max_dim=2):
    """Returns True if first element of the incompatible
    array type is boolean.
    """
    if max_dim < 1:
        return None
    try:
        first = next(iter(idx), None)
    except TypeError:
        return None
    if isinstance(first, bool):
        return True
    return _first_element_bool(first, max_dim-1)


def _compatible_boolean_index(idx):
    """Returns a boolean index array that can be converted to
    integer array. Returns None if no such array exists.
    """
    # Presence of attribute `ndim` indicates a compatible array type.
    if hasattr(idx, 'ndim') or _first_element_bool(idx):
        return _maybe_bool_ndarray(idx)
    return None


def _boolean_index_to_array(idx):
    if idx.ndim > 1:
        raise IndexError('invalid index shape')
    return cupy.where(idx)[0]
