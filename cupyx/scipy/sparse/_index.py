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
_bool_scalar_types = (bool, numpy.bool_)


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


def _get_csr_submatrix_major_axis(Ax, Aj, Ap, start, stop):
    """Return a submatrix of the input sparse matrix by slicing major axis.

    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        start (int): starting index of major axis
        stop (int): ending index of major axis

    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array of output sparse matrix

    """
    Ap = Ap[start:stop + 1]
    start_offset, stop_offset = int(Ap[0]), int(Ap[-1])
    Bp = Ap - start_offset
    Bj = Aj[start_offset:stop_offset]
    Bx = Ax[start_offset:stop_offset]

    return Bx, Bj, Bp


def _get_csr_submatrix_minor_axis(Ax, Aj, Ap, start, stop):
    """Return a submatrix of the input sparse matrix by slicing minor axis.

    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        start (int): starting index of minor axis
        stop (int): ending index of minor axis

    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array of output sparse matrix

    """
    mask = (start <= Aj) & (Aj < stop)
    mask_sum = cupy.empty(Aj.size + 1, dtype=Aj.dtype)
    mask_sum[0] = 0
    mask_sum[1:] = mask
    cupy.cumsum(mask_sum, out=mask_sum)
    Bp = mask_sum[Ap]
    Bj = Aj[mask] - start
    Bx = Ax[mask]

    return Bx, Bj, Bp


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


def _csr_row_index(Ax, Aj, Ap, rows):
    """Populate indices and data arrays from the given row index
    Args:
        Ax (cupy.ndarray): data array from input sparse matrix
        Aj (cupy.ndarray): indices array from input sparse matrix
        Ap (cupy.ndarray): indptr array from input sparse matrix
        rows (cupy.ndarray): index array of rows to populate
    Returns:
        Bx (cupy.ndarray): data array of output sparse matrix
        Bj (cupy.ndarray): indices array of output sparse matrix
        Bp (cupy.ndarray): indptr array for output sparse matrix
    """
    row_nnz = cupy.diff(Ap)
    Bp = cupy.empty(rows.size + 1, dtype=Ap.dtype)
    Bp[0] = 0
    cupy.cumsum(row_nnz[rows], out=Bp[1:])
    nnz = int(Bp[-1])

    out_rows = _csr_indptr_to_coo_rows(nnz, Bp)

    Bj, Bx = _csr_row_index_ker(out_rows, rows, Ap, Aj, Ax, Bp)
    return Bx, Bj, Bp


def _csr_indptr_to_coo_rows(nnz, Bp):
    out_rows = cupy.empty(nnz, dtype=numpy.int32)

    # Build a COO row array from output CSR indptr.
    # Calling backend cusparse API directly to avoid
    # constructing a whole COO object.
    handle = device.get_cusparse_handle()
    cusparse.xcsr2coo(
        handle, Bp.data.ptr, nnz, Bp.size-1, out_rows.data.ptr,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)

    return out_rows


def _select_last_indices(i, j, x, idx_dtype):
    """Find the unique indices for each row and keep only the last"""
    i = cupy.asarray(i, dtype=idx_dtype)
    j = cupy.asarray(j, dtype=idx_dtype)

    stacked = cupy.stack([j, i])
    order = cupy.lexsort(stacked).astype(idx_dtype)

    indptr_inserts = i[order]
    indices_inserts = j[order]
    data_inserts = x[order]

    mask = cupy.ones(indptr_inserts.size, dtype='bool')
    _unique_mask_kern(indptr_inserts, indices_inserts, order, mask,
                      size=indptr_inserts.size-1)

    return indptr_inserts[mask], indices_inserts[mask], data_inserts[mask]


_insert_many_populate_arrays = core.ElementwiseKernel(
    '''raw I insert_indices, raw T insert_values, raw I insertion_indptr,
        raw I Ap, raw I Aj, raw T Ax, raw I Bp''',
    'raw I Bj, raw T Bx', '''

        const I input_row_start = Ap[i];
        const I input_row_end = Ap[i+1];
        const I input_count = input_row_end - input_row_start;

        const I insert_row_start = insertion_indptr[i];
        const I insert_row_end = insertion_indptr[i+1];
        const I insert_count = insert_row_end - insert_row_start;

        I input_offset = 0;
        I insert_offset = 0;

        I output_n = Bp[i];

        I cur_existing_index = -1;
        T cur_existing_value = -1;

        I cur_insert_index = -1;
        T cur_insert_value = -1;

        if(input_offset < input_count) {
            cur_existing_index = Aj[input_row_start+input_offset];
            cur_existing_value = Ax[input_row_start+input_offset];
        }

        if(insert_offset < insert_count) {
            cur_insert_index = insert_indices[insert_row_start+insert_offset];
            cur_insert_value = insert_values[insert_row_start+insert_offset];
        }


        for(I jj = 0; jj < input_count + insert_count; jj++) {

            // if we have both available, use the lowest one.
            if(input_offset < input_count &&
               insert_offset < insert_count) {

                if(cur_existing_index < cur_insert_index) {
                    Bj[output_n] = cur_existing_index;
                    Bx[output_n] = cur_existing_value;

                    ++input_offset;

                    if(input_offset < input_count) {
                        cur_existing_index = Aj[input_row_start+input_offset];
                        cur_existing_value = Ax[input_row_start+input_offset];
                    }


                } else {
                    Bj[output_n] = cur_insert_index;
                    Bx[output_n] = cur_insert_value;

                    ++insert_offset;
                    if(insert_offset < insert_count) {
                        cur_insert_index =
                            insert_indices[insert_row_start+insert_offset];
                        cur_insert_value =
                            insert_values[insert_row_start+insert_offset];
                    }
                }

            } else if(input_offset < input_count) {
                Bj[output_n] = cur_existing_index;
                Bx[output_n] = cur_existing_value;

                ++input_offset;
                if(input_offset < input_count) {
                    cur_existing_index = Aj[input_row_start+input_offset];
                    cur_existing_value = Ax[input_row_start+input_offset];
                }

            } else {
                    Bj[output_n] = cur_insert_index;
                    Bx[output_n] = cur_insert_value;

                    ++insert_offset;
                    if(insert_offset < insert_count) {
                        cur_insert_index =
                            insert_indices[insert_row_start+insert_offset];
                        cur_insert_value =
                            insert_values[insert_row_start+insert_offset];
                    }
            }

            output_n++;
        }
    ''', 'csr_copy_existing_indices_kern', no_return=True)


# Create a filter mask based on the lowest value of order
_unique_mask_kern = core.ElementwiseKernel(
    '''raw I rows, raw I cols, raw I order''',
    '''raw bool mask''',
    """
    I cur_row = rows[i];
    I next_row = rows[i+1];

    I cur_col = cols[i];
    I next_col = cols[i+1];

    I cur_order = order[i];
    I next_order = order[i+1];

    if(cur_row == next_row && cur_col == next_col) {
        if(cur_order < next_order)
            mask[i] = false;
        else
            mask[i+1] = false;
    }
    """, no_return=True
)


def _csr_sample_values(n_row, n_col,
                       Ap, Aj, Ax,
                       Bi, Bj, not_found_val=0):
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

    return _csr_sample_values_kern(n_row, n_col,
                                   Ap, Aj, Ax,
                                   Bi, Bj,
                                   not_found_val,
                                   size=Bi.size)


_csr_sample_values_kern = core.ElementwiseKernel(
    '''I n_row, I n_col, raw I Ap, raw I Aj, raw T Ax,
    raw I Bi, raw I Bj, I not_found_val''',
    'raw T Bx', '''
    const I j = Bi[i]; // sample row
    const I k = Bj[i]; // sample column
    const I row_start = Ap[j];
    const I row_end   = Ap[j+1];
    T x = 0;
    bool val_found = false;
    for(I jj = row_start; jj < row_end; jj++) {
        if (Aj[jj] == k) {
            x += Ax[jj];
            val_found = true;
        }
    }
    Bx[i] = val_found ? x : not_found_val;
''', 'csr_sample_values_kern')


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

    def __setitem__(self, key, x):
        row, col = self._parse_indices(key)

        if isinstance(row, _int_scalar_types) and\
                isinstance(col, _int_scalar_types):
            x = cupy.asarray(x, dtype=self.dtype)
            if x.size != 1:
                raise ValueError('Trying to assign a sequence to an item')
            self._set_intXint(row, col, x.flat[0])
            return

        if isinstance(row, slice):
            row = cupy.arange(*row.indices(self.shape[0]))[:, None]
        else:
            row = cupy.atleast_1d(row)

        if isinstance(col, slice):
            col = cupy.arange(*col.indices(self.shape[1]))[None, :]
            if row.ndim == 1:
                row = row[:, None]
        else:
            col = cupy.atleast_1d(col)

        i, j = cupy.broadcast_arrays(row, col)
        if i.shape != j.shape:
            raise IndexError('number of row and column indices differ')

        if isspmatrix(x):
            if i.ndim == 1:
                # Inner indexing, so treat them like row vectors.
                i = i[None]
                j = j[None]
            broadcast_row = x.shape[0] == 1 and i.shape[0] != 1
            broadcast_col = x.shape[1] == 1 and i.shape[1] != 1
            if not ((broadcast_row or x.shape[0] == i.shape[0]) and
                    (broadcast_col or x.shape[1] == i.shape[1])):
                raise ValueError('shape mismatch in assignment')
            if x.size == 0:
                return
            x = x.tocoo(copy=True)
            x.sum_duplicates()
            self._set_arrayXarray_sparse(i, j, x)
        else:
            # Make x and i into the same shape
            x = cupy.asarray(x, dtype=self.dtype)
            x, _ = cupy.broadcast_arrays(x, i)
            if x.size == 0:
                return
            x = x.reshape(i.shape)
            self._set_arrayXarray(i, j, x)

    def _is_scalar(self, index):
        if isinstance(index, (cupy.ndarray, numpy.ndarray)) and \
                index.ndim == 0 and index.size == 1:
            return True
        return False

    def _parse_indices(self, key):
        M, N = self.shape
        row, col = _unpack_index(key)

        if self._is_scalar(row):
            row = row.item()
        if self._is_scalar(col):
            col = col.item()

        # Scipy calls sputils.isintlike() rather than
        # isinstance(x, _int_scalar_types). Comparing directly to int
        # here to minimize the impact of nested exception catching

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
