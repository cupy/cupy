"""Indexing mixin for sparse matrix classes.
"""
import math
import cupy
from .sputils import isintlike

try:
    INT_TYPES = (int, long, cupy.integer)
except NameError:
    # long is not defined in Python3
    INT_TYPES = (int, cupy.integer)


def _broadcast_arrays(a, b):
    """
    Same as cupy.broadcast_arrays(a, b) but old writeability rules.
    NumPy >= 1.17.0 transitions broadcast_arrays to return
    read-only arrays. Set writeability explicitly to avoid warnings.
    Retain the old writeability rules, as our Cython code assumes
    the old behavior.
    """
    x, y = cupy.broadcast_arrays(a, b)
    x.flags.writeable = a.flags.writeable
    y.flags.writeable = b.flags.writeable
    return x, y


bin_col_offsets_ker = cupy.RawKernel("""
    extern "C" __global__
    void bin_col_offsets_ker_str(int n_idx, int *col_idxs, int *col_offsets) {
                                  
        // Get the index of the thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(i > n_idx) return;
        col_offsets[i]++;
""", "bin_col_offsets_ker_str")


def bin_col_offsets(n_idx, col_idsx, col_offsets, tpb=32):
    grid = math.ceil(n_idx / tpb)
    bin_col_offsets_ker((grid,), (tpb,), (n_idx, col_idsx, col_offsets))

csr_column_index1_ker = cupy.RawKernel("""
    extern "C" __global__
    void csr_column_index1_ker_str(int n_row, 
                                 int *col_offsets,
                                 int *Ap,
                                 int *Aj,
                                 int *Bp) {
                                
        // Get the index of the thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(i > n_row) return;

        int new_col_size = 0;
        
        for(int jj = Ap[i]; jj < Ap[i+1]; jj++) 
            new_col_size += col_offsets[Aj[jj]];
        
        Bp[i+1] = new_col_size;
}
""", "csr_column_index1_ker_str")


def csr_column_index1_degree(n_row, col_offsets, Ap, Aj, Bp, tpb=32):
    grid = math.ceil(n_row / tpb)
    csr_column_index1_ker((grid,), (tpb,), (n_row, col_offsets, Ap, Aj, Bp))


def csr_column_index1(n_idx, col_idxs, n_row, n_col,
                          indptr, indices, offsets, new_indptr):
    bin_col_offsets(n_idx, col_idxs, offsets)
    csr_column_index1_degree(n_row, offsets, indptr, indices, new_indptr)

    cupy.cumsum(offsets, out=offsets)
    cupy.cumsum(new_indptr, out=new_indptr)


get_csr_index2_ker = cupy.RawKernel("""
    extern "C" __global__
    void get_csr_index2_ker_str(int *col_order,
                                int *col_offsets,
                                const int *Ap,
                                const int *Aj,
                                const float *Ax,
                                int n_row,
                                int *Bp,
                                int *Bj,
                                float *Bx) {
                                
    // Get the index of the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i > n_row) return;
    

    int n = Bp[i];
    
    for(int jj = Ap[i]; jj < Ap[i+1]; i++) {
        const int j = Aj[jj];
        
        const int offset = col_offsets[j];
        const int prev_offset = j == 0 ? 0 : col_offsets[j-1];
        if (offset != prev_offset) {
            const float v = Ax[jj];
            for(int k = prev_offset; k < offset; k++){
                Bj[n] = col_order[k];
                Bx[n] = v;
                n++;
            }
        }
    }
}
""", "get_csr_index2_ker_str")


def csr_column_index2(out_rows, col_order, col_offsets, nnz,
                   Ap, Aj, Ax, Bp, Bj, Bx, tpb=32):

    grid = math.ceil(out_rows / tpb)
    get_csr_index2_ker((grid,), (tpb,), (col_order, col_offsets,
                                         Ap, Aj, Ax, len(Ap), Bp, Bj, Bx))


def get_csr_submatrix(n_row, n_col, indptr, indices, data,
                      start_maj, stop_maj, start_min, stop_min):

    # We first compute the degree, then use it to compute the indptr
    new_n_row = stop_maj - start_maj

    new_indptr = cupy.zeros((new_n_row+1,), dtype=indptr.dtype)

    get_csr_submatrix_degree(new_n_row, indptr, indices,
                             start_maj, stop_maj,
                             start_min, stop_min, new_indptr)

    cupy.cumsum(new_indptr, out=new_indptr)

    new_nnz = new_indptr[-1]

    new_indices = cupy.zeros((new_nnz,), dtype=indices.dtype)
    new_data = cupy.zeros((new_nnz,), dtype=data.dtype)

    get_csr_submatrix_cols_data(new_n_row, indptr, indices, data,
                                start_maj, stop_maj,
                                start_min, stop_min,
                                new_indptr, new_indices, new_data)

    return new_indptr, new_indices, new_data


get_csr_submatrix_degree_ker = cupy.RawKernel("""
    extern "C" __global__
    void get_csr_submatrix_degree_kernel(const int *Ap,
                                         const int *Aj,
                                         const int ir0,
                                         const int ir1,
                                         const int ic0,
                                         const int ic1,
                                         int *Bp) {
                                  
        // Get the index of the thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(i > new_n_row) return;
        
        int row_start = Ap[ir0+i];
        int row_end = Ap[ir0+i+1];
        
        int row_count = 0;
        for int jj = row_start; jj < row_end; jj++) {
            if((Aj[jj] >= ic0) && (Aj[jj] < ic1)) 
                row_count++;
        }
        
        Bp[i+1] = row_count;
    }
""", "get_csr_submatrix_degree_kernel")


def get_csr_submatrix_degree(new_rows, Ap, Aj, ir0, ir1,
                             ic0, ic1, Bp, tpb=32):

    """
    Invokes get_csr_submatrix_degree_ker with the given inputs
    """

    grid = math.ceil(new_rows / tpb)
    get_csr_submatrix_degree_ker((grid,), (tpb,),
                                 (Ap, Aj, ir0, ir1,
                                  ic0, ic1, Bp))


get_csr_submatrix_cols_data_ker = cupy.RawKernel("""
    extern "C" __global__
    void get_csr_submatrix_cols_data(const int *Ap,
                                     const int *Aj,
                                     const float *Ax,
                                     const int ir0,
                                     const int ir1,
                                     const int ic0,
                                     const int ic1,
                                     int *Bp,
                                     int *Bj,
                                     float *Bx) {

        // Get the index of the thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        I row_start = Ap[ir0+i];
        I row_end   = Ap[ir0+i+1];

        int kk = Bp[i];

        for(int jj = row_start; jj < row_end; jj++) {
            if ((Aj[jj] >= ic0) && (Aj[jj] < ic1)) {
                (*Bj)[kk] = Aj[jj] - ic0;
                (*Bx)[kk] = Ax[jj];
                kk++;
            }
        }
""", "get_csr_submatrix_cols_data")


def get_csr_submatrix_cols_data(new_rows,
                                Ap, Aj, Ax,
                                ir0, ir1,
                                ic0, ic1,
                                Bp, Bj, Bx,
                                tpb=32):

    grid = math.ceil(new_rows/tpb)
    get_csr_submatrix_cols_data_ker((grid,), (tpb,),
                                    (Ap, Aj, Ax,
                                     ir0, ir1,
                                     ic0, ic1,
                                     Bp, Bj, Bx))


csr_row_index_ker = cupy.RawKernel("""
    extern "C" __global__
    void csr_row_index_ker_str(const int n_row_idx,
                               const int *rows,
                               const int *Ap,
                               const int *Aj,
                               const float *Ax,
                               const int *Bp,
                               const int *Bj,
                               const float *Bx) {
                               
        // Get the index of the thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(i > n_row_idx) return;
        
        const int row = rows[i];
        const int row_start = Ap[row];
        const int row_end = Ap[row+1];
        
        const int out_row_idx = Bp[i];

        for(int j = row_start; j < row_end; j++) {
            Bj[out_row_idx] = Aj[j];
            Bx[out_row_idx] = Ax[j];
            out_row_idx++;
        }

""", "csr_row_index_ker_str")


def csr_row_index(n_row_idx, rows,
                  Ap, Aj, Ax,
                  Bp, Bj, Bx, tpb=32):

    grid = math.ceil(n_row_idx / tpb)

    csr_row_index_ker((grid,), (tpb,),
                      (n_row_idx, rows, Ap, Aj, Ax,
                       Bp, Bj, Bx))



# @TODO: Port this to CUDA
"""
/*
 * Sample the matrix at specific locations
 *
 * Determine the matrix value for each row,col pair
 *    Bx[n] = A(Bi[n],Bj[n])
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *   I  n_samples     - number of samples
 *   I  Bi[N]         - sample rows
 *   I  Bj[N]         - sample columns
 *
 * Output Arguments:
 *   T  Bx[N]         - sample values
 *
 * Note:
 *   Output array Bx must be preallocated
 *
 *   Complexity: varies
 *
 *   TODO handle other cases with asymptotically optimal method
 *
 */
template <class I, class T>
void csr_sample_values(const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                       const T Ax[],
                       const I n_samples,
                       const I Bi[],
                       const I Bj[],
                             T Bx[])
{
    // ideally we'd do the following
    // Case 1: A is canonical and B is sorted by row and column
    //   -> special purpose csr_binop_csr() (optimized form)
    // Case 2: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) > log(num_samples)
    //   -> do binary searches for each sample
    // Case 3: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) < log(num_samples)
    //   -> sort B by row and column and use Case 1
    // Case 4: A is not canonical and num_samples ~ nnz
    //   -> special purpose csr_binop_csr() (general form)
    // Case 5: A is not canonical and num_samples << nnz
    //   -> do linear searches for each sample

    const I nnz = Ap[n_row];

    const I threshold = nnz / 10; // constant is arbitrary

    if (n_samples > threshold && csr_has_canonical_format(n_row, Ap, Aj))
    {
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // sample row
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // sample column

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            if (row_start < row_end)
            {
                const I offset = std::lower_bound(Aj + row_start, Aj + row_end, j) - Aj;

                if (offset < row_end && Aj[offset] == j)
                    Bx[n] = Ax[offset];
                else
                    Bx[n] = 0;
            }
            else
            {
                Bx[n] = 0;
            }

        }
    }
    else
    {
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // sample row
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // sample column

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            T x = 0;

            for(I jj = row_start; jj < row_end; jj++)
            {
                if (Aj[jj] == j)
                    x += Ax[jj];
            }

            Bx[n] = x;
        }

    }
}
"""


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
                return self._get_columnXarray(row[:,0], col.ravel())

        # The only remaining case is inner (fancy) indexing
        row, col = _broadcast_arrays(row, col)
        if row.shape != col.shape:
            raise IndexError('number of row and column indices differ')
        if row.size == 0:
            return self.__class__(cupy.atleast_2d(row).shape, dtype=self.dtype)
        return self._get_arrayXarray(row, col)

    def __setitem__(self, key, x):
        row, col = self._validate_indices(key)

        if isinstance(row, INT_TYPES) and isinstance(col, INT_TYPES):
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

        i, j = _broadcast_arrays(row, col)
        if i.shape != j.shape:
            raise IndexError('number of row and column indices differ')

        from .base import isspmatrix
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
            x, _ = _broadcast_arrays(x, i)
            if x.size == 0:
                return
            x = x.reshape(i.shape)
            self._set_arrayXarray(i, j, x)

    def _validate_indices(self, key):
        M, N = self.shape
        row, col = _ucupyack_index(key)

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
            x = cupy.asarray(idx)
        except (ValueError, TypeError, MemoryError):
            raise IndexError('invalid index')

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')

        if x.size == 0:
            return x

        # Check bounds
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError('index (%d) out of range' % max_indx)

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError('index (%d) out of range' % min_indx)
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    def getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.
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
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)


def _ucupyack_index(index):
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

    # TODO: Deprecate this multiple-ellipsis handling,
    #       as numpy no longer supports it.

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
