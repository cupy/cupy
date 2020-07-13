"""Indexing mixin for sparse matrix classes.
"""
import math
from .sputils import isintlike

from numpy import integer
from numpy import dtype

import cupy
import cupyx
from cupy import core


INT_TYPES = (int, integer)

_int32_dtype = dtype('int32')
_int64_dtype = dtype('int64')
_float32_dtype = dtype('float32')
_float64_dtype = dtype('float64')
_complex64_dtype = dtype('complex64')
_complex128_dtype = dtype('complex128')

_supported_types = {
    _int32_dtype: "int",
    _int64_dtype: "long long",
    _float32_dtype: "float",
    _float64_dtype: "double",
    _complex64_dtype: "complex<float>",
    _complex128_dtype: "complex<double>"
}

_module_options = ('--std=c++11',)


def _build_name_expressions(types, kernel_name):

    def parse_types(t):
        if isinstance(types[0], tuple):
            return ",".join([_supported_types[tu] for tu in t])
        else:
            return _supported_types[t]

    return {t: "%s<%s>" % (kernel_name, parse_types(t)) for t in types}


def _broadcast_arrays(a, b):
    """
    Same as cupy.broadcast_arrays(a, b) but old writeability rules.
    NumPy >= 1.17.0 transitions broadcast_arrays to return
    read-only arrays. Set writeability explicitly to avoid warnings.
    Retain the old writeability rules, as our Cython code assumes
    the old behavior.
    """
    x, y = cupy.broadcast_arrays(a, b)

    # Writeable doesn't seem to exist on flags in cupy
    # x.flags.writeable = a.flags.writeable
    # y.flags.writeable = b.flags.writeable
    return x, y


_csr_column_index2_order_types = \
    _build_name_expressions([_int32_dtype, _int64_dtype],
                            '_csr_column_index2_order')
_csr_column_index2_order_ker = core.RawModule(code="""
    template<typename I>
    __global__
    void _csr_column_index2_order(const I *col_order,
                                  const I n,
                                  I *out_index) {

    // Get the index of the thread
    I i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
        out_index[col_order[i]] = col_order[i+1];
}
""", options=_module_options, name_expressions=tuple(
    _csr_column_index2_order_types.values()
))


def _csr_column_index2_order(col_order, tpb=1024):
    """Index an array of argsorted indices for looking up
    locations of duplicates.

    Args
       col_order : array of argsorted indices from index

    Returns
        out_order : new array where each element points to
                    its next potential duplicate in the index
    """
    grid = math.ceil(col_order.size-1 / tpb)
    kernel = _csr_column_index2_order_ker.get_function(
        _csr_column_index2_order_types[col_order.dtype]
    )
    out_order = cupy.empty_like(col_order)
    kernel((grid,), (tpb,), (col_order, col_order.size-1, out_order))

    return out_order


def _csr_column_index1_indptr(unique_idxs, sort_idxs, col_counts,
                              Ap, Aj):
    """Construct output indptr by counting column indices
    in input matrix for each row.

    Args
        unique_idxs : Unique set of indices sorted in ascending order
        sort_idxs : Indices sorted to preserve original order of unique_idxs
        col_counts : Number of times each unique index occurs in Aj
        Ap : indptr array of input sparse matrix
        Aj : indices array of input sparse matrix

    Returns
        Bp : Output indptr
        Aj_mask : Input indices array with all cols not matching the
                  index masked out with -1.
    """
    out_col_sum = cupy.zeros((Aj.size+1,), dtype=col_counts.dtype)

    index = cupy.argsort(unique_idxs)
    sorted_index = cupy.searchsorted(unique_idxs, Aj)

    yindex = cupy.take(index, sorted_index)
    mask = unique_idxs[yindex] == Aj

    # TODO(cjnolet): Find a more efficient approach for this.
    idxs_adj = _csr_column_inv_idx(unique_idxs)
    out_col_sum[1:][mask] = col_counts[idxs_adj[Aj[mask]]]

    Aj_mask = out_col_sum[1:].copy()
    Aj_mask[Aj_mask == 0] = -1

    Aj_mask[Aj_mask > 0] = Aj[Aj_mask > 0]
    Aj_mask[Aj_mask > 0] = cupy.searchsorted(
        unique_idxs, Aj_mask[Aj_mask > 0])

    Aj_mask[Aj_mask >= 0] = sort_idxs[Aj_mask[Aj_mask >= 0]]

    cupy.cumsum(out_col_sum, out=out_col_sum)
    Bp = out_col_sum[Ap]
    Bp[1:] -= Bp[:-1]
    cupy.cumsum(Bp, out=Bp)

    return Bp, Aj_mask


def _csr_column_index1(col_idxs, Ap, Aj):
    """Construct indptr and components for populating indices and data of
    output sparse array

    Args
        col_idxs : column indices to index from input indices
        Ap : indptr of input sparse matrix
        Aj : indices of input sparse matrix

    Returns
        Bp : indptr of output sparse matrix
        Aj_mask : Input indices array with all cols not matching the index
                  index masked out with -1.
        col_counts : Number of times each unique index occurs in Aj
        sort_idxs : Indices sorted to preserve original order of idxs
    """

    idx_map, sort_idxs = cupy.unique(col_idxs, return_index=True)
    sort_idxs = sort_idxs.astype(idx_map.dtype)
    idxs = cupy.searchsorted(idx_map, col_idxs)

    col_counts = cupy.zeros(idx_map.size, dtype=col_idxs.dtype)
    cupyx.scatter_add(col_counts, idxs, 1)

    Bp, Aj_mask = _csr_column_index1_indptr(
        idx_map, sort_idxs, col_counts, Ap, Aj)

    return Bp, Aj_mask, col_counts, sort_idxs


_get_csr_index2_ker_types = _build_name_expressions(
    [(_int32_dtype, _float32_dtype), (_int32_dtype, _float64_dtype),
     (_int32_dtype, _complex64_dtype), (_int32_dtype, _complex128_dtype),
     (_int64_dtype, _float32_dtype), (_int64_dtype, _float64_dtype),
     (_int64_dtype, _complex64_dtype), (_int64_dtype, _complex128_dtype)],
    'csr_index2_ker')
_get_csr_index2_ker = core.RawModule(code="""
    #include <cupy/complex.cuh>
    template<typename I, typename T>
    __global__
    void csr_index2_ker(I *idxs,
                        I *col_counts,
                        I *col_order,
                        const I *Ap,
                        const I *Aj,
                        const T *Ax,
                        I n_row,
                        I *Bp,
                        I *Bj,
                        T *Bx) {

    // Get the index of the thread
    I i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_row) {

        I n = Bp[i];

        // loop through columns in current row
        for(int jj = Ap[i]; jj < Ap[i+1]; jj++) {
            I col = Aj[jj];  // current column
            if(col != -1) {
                const T v = Ax[jj];
                const I counts = col_counts[idxs[col]];
                for(int l = 0; l < counts; l++) {
                    if(l > 0)
                        col = col_order[col];
                    Bj[n] = col;
                    Bx[n] = v;
                    n++;
                }
            }
        }
    }
}
""", options=_module_options, name_expressions=tuple(
    _get_csr_index2_ker_types.values()))


_csr_column_inv_idx_types = _build_name_expressions(
    [_int32_dtype, _int64_dtype], "_csr_column_inv_idx")
_csr_column_inv_idx_ker = core.RawModule(code="""
    template<typename I>
    __global__
    void _csr_column_inv_idx(const I *idxs,
                                const I n,
                                I *out_index) {

    // Get the index of the thread
    I i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
        out_index[idxs[i]] = i;
}
""", options=_module_options, name_expressions=tuple(
    _csr_column_inv_idx_types.values()
))


def _csr_column_inv_idx(idxs, tpb=1024):
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

    grid = math.ceil(idxs.size / tpb)
    kernel = _csr_column_inv_idx_ker.get_function(
        _csr_column_inv_idx_types[idxs.dtype]
    )
    kernel((grid,), (tpb,), (idxs, idxs.size, idxs_adj))

    return idxs_adj


def _csr_column_index2(col_order,
                       col_counts,
                       sort_idxs,
                       Ap, Aj_mask, Ax,
                       Bp, tpb=1024):
    """Populate indices and data arrays from column index

    Args
        col_order : argsort order of column index
        col_counts : counts of each unique index item from Aj_mask
        sort_idxs : Indices of unique index columns sorted to preserve
                    original order
        Ap : indptr array of input sparse array
        Aj_mask : Input indices array with all cols not matching the
                  index masked out with -1.
        Ax : data array of input sparse matrix
        Bp : indptr array of output sparse matrix
        tpb : Threads per block for populating indices and data

    Returns
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """

    new_nnz = Bp[-1].item()

    Bj = cupy.zeros(new_nnz, dtype=Aj_mask.dtype)
    Bx = cupy.zeros(new_nnz, dtype=Ax.dtype)

    col_order = _csr_column_index2_order(col_order)
    idxs_adj = _csr_column_inv_idx(sort_idxs)

    grid = math.ceil((Bp.size-1) / tpb)
    func = _get_csr_index2_ker_types[(Ap.dtype, Bx.dtype)]
    kernel = _get_csr_index2_ker.get_function(func)
    kernel((grid,), (tpb,),
           (idxs_adj,
            col_counts,
            col_order,
            Ap, Aj_mask, Ax, Ap.size - 1, Bp, Bj, Bx))

    return Bj, Bx


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

    Bp = _get_csr_submatrix_degree(Ap, Aj,
                                   start_maj, stop_maj,
                                   start_min, stop_min)

    cupy.cumsum(Bp, out=Bp)

    # Computing this once to minimize dev->host sync
    new_nnz = Bp[-1].item()
    if new_nnz > 0:
        Bj, Bx = _get_csr_submatrix_cols_data(Ap, Aj, Ax,
                                              start_maj, stop_maj,
                                              start_min, stop_min,
                                              new_nnz, Bp)
    else:
        Bj = cupy.empty(new_nnz, dtype=Aj.dtype)
        Bx = cupy.empty(new_nnz, dtype=Ax.dtype)

    return Bp, Bj, Bx


_get_csr_submatrix_degree_ker_types = _build_name_expressions(
    [_int32_dtype, _int64_dtype], 'csr_submatrix_degree_ker')
_get_csr_submatrix_degree_ker = core.RawModule(code="""
    template<typename I>
    __global__
    void csr_submatrix_degree_ker(const I *Ap,
                                  const I *Aj,
                                  const I ir0,
                                  const I ir1,
                                  const I ic0,
                                  const I ic1,
                                  I *Bp) {

        // Get the index of the thread
        I i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < (ir1-ir0)) {

            const I row_start = Ap[ir0+i];
            const I row_end = Ap[ir0+i+1];

            I row_count = 0;
            for(I jj = row_start; jj < row_end; jj++) {
                I col = Aj[jj];
                if((col >= ic0) && (col < ic1))
                    row_count++;
            }

            if(row_count > 0)
                Bp[i+1] = row_count;
        }
    }
""", options=_module_options, name_expressions=tuple(
    _get_csr_submatrix_degree_ker_types.values()))


def _get_csr_submatrix_degree(Ap, Aj, ir0, ir1,
                              ic0, ic1, tpb=1024):
    """
    Compute indptr array of output sparse matrix after
    slicing major and minor axes of input sparse matrix.

    Args
        Ap : indptr of input sparse matrix
        Aj : indices of input sparse matrix
        ir0 : starting index of major axis
        ir1 : ending index of major axis
        ic0 : starting index of minor axis
        ic1 : ending index of minor axis
        tpb : threads per block of degree computation kernel

    Returns
        Bp : indptr of output sparse matrix
    """
    new_n_row = ir1 - ir0
    Bp = cupy.zeros(new_n_row + 1, dtype=Ap.dtype)

    grid = math.ceil((ir1-ir0) / tpb)
    kernel = _get_csr_submatrix_degree_ker.get_function(
        _get_csr_submatrix_degree_ker_types[Ap.dtype]
    )
    kernel((grid,), (tpb,),
           (Ap, Aj, ir0, ir1,
            ic0, ic1, Bp))

    return Bp


_get_csr_submatrix_cols_data_ker_types = _build_name_expressions(
    [(_int32_dtype, _float32_dtype), (_int32_dtype, _float64_dtype),
     (_int32_dtype, _complex64_dtype), (_int32_dtype, _complex128_dtype),
     (_int64_dtype, _float32_dtype), (_int64_dtype, _float64_dtype),
     (_int64_dtype, _complex64_dtype), (_int64_dtype, _complex128_dtype)],
    'get_csr_submatrix_cols_data')
_get_csr_submatrix_cols_data_ker = core.RawModule(code="""
    #include <cupy/complex.cuh>
    template<typename I, typename T>
    __global__
    void get_csr_submatrix_cols_data(const I *Ap,
                                     const I *Aj,
                                     const T *Ax,
                                     const I ir0,
                                     const I ir1,
                                     const I ic0,
                                     const I ic1,
                                     const I *Bp,
                                     I *Bj,
                                     T *Bx) {

        // Get the index of the thread
        I i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < (ir1-ir0)) {

            I row_start = Ap[ir0+i];
            I row_end   = Ap[ir0+i+1];

            I kk = Bp[i];

            for(I jj = row_start; jj < row_end; jj++) {
                I col = Aj[jj];
                if ((col >= ic0) && (col < ic1)) {
                    Bj[kk] = col - ic0;
                    Bx[kk] = Ax[jj];
                    kk++;
                }
            }
        }
    }
""", options=_module_options, name_expressions=tuple(
    _get_csr_submatrix_cols_data_ker_types.values()))


def _get_csr_submatrix_cols_data(Ap, Aj, Ax,
                                 ir0, ir1,
                                 ic0, ic1,
                                 new_nnz, Bp,
                                 tpb=1024):
    """Populate indices and data arrays for submatrix slicing

    Args
        Ap : indptr array from input sparse matrix
        Aj : indices array from input sparse matrix
        Ax : data array from input sparse matrix
        ir0 : starting index for major axis
        ir1 : ending index for major axis
        ic0 : starting index for minor axis
        ic1 : ending index for minor axis
        new_nnz : nnz for output sparse matrix
        Bp : indptr array for output sparse matrix
        tpb : threads per block for cuda kernel to populate indices/data

    Returns
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """

    Bj = cupy.zeros(new_nnz, dtype=Aj.dtype)
    Bx = cupy.zeros(new_nnz, dtype=Ax.dtype)

    grid = math.ceil((ir1-ir0)/tpb)
    kernel = _get_csr_submatrix_cols_data_ker.get_function(
        _get_csr_submatrix_cols_data_ker_types[(Ap.dtype, Ax.dtype)]
    )
    kernel((grid,), (tpb,),
           (Ap, Aj, Ax,
            ir0, ir1,
            ic0, ic1,
            Bp, Bj, Bx))

    return Bj, Bx


_csr_row_index_ker_types = _build_name_expressions(
    [(_int32_dtype, _float32_dtype), (_int32_dtype, _float64_dtype),
     (_int32_dtype, _complex64_dtype), (_int32_dtype, _complex128_dtype),
     (_int64_dtype, _float32_dtype), (_int64_dtype, _float64_dtype),
     (_int64_dtype, _complex64_dtype), (_int64_dtype, _complex128_dtype)],
    'csr_row_index_ker')
_csr_row_index_ker = core.RawModule(code="""
    #include <cupy/complex.cuh>
    template<typename I, typename T>
    __global__
    void csr_row_index_ker(const I n_row_idx,
                           const I *rows,
                           const I *Ap,
                           const I *Aj,
                           const T *Ax,
                           const I *Bp,
                           I *Bj,
                           T *Bx) {

        // Get the index of the thread
        I i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n_row_idx) {

            I row = rows[i];
            I row_start = Ap[row];
            I row_end = Ap[row+1];

            I out_row_idx = Bp[i];

            // Copy columns
            for(I j = row_start; j < row_end; j++) {
                Bj[out_row_idx] = Aj[j];
                Bx[out_row_idx] = Ax[j];
                out_row_idx++;
            }
        }
    }

""", options=_module_options, name_expressions=tuple(
    _csr_row_index_ker_types.values()))


def _csr_row_index(rows,
                   Ap, Aj, Ax,
                   Bp, tpb=1024):
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

    grid = math.ceil(rows.size / tpb)

    nnz = Bp[-1].item()
    Bj = cupy.empty(nnz, dtype=Aj.dtype)
    Bx = cupy.empty(nnz, dtype=Ax.dtype)

    kernel = _csr_row_index_ker.get_function(
        _csr_row_index_ker_types[(Ap.dtype, Ax.dtype)]
    )
    kernel((grid,), (tpb,),
           (rows.size, rows,
            Ap, Aj, Ax,
            Bp, Bj, Bx))

    return Bj, Bx


def _csr_sample_values(n_row, n_col,
                       Ap, Aj, Ax,
                       Bi, Bj, tpb=1024):
    """Populate data array for a set of rows and columns

    Args
        n_row : total number of rows in input array
        n_col : total number of columns in input array
        Ap : indptr array for input sparse matrix
        Aj : indices array for input sparse matrix
        Ax : data array for input sparse matrix
        Bi : array of rows to extract from input sparse matrix
        Bj : array of columns to extract from input sparse matrix
        tpb : threads per block for kernel

    Returns
        Bx : data array for output sparse matrix
    """

    n_samples = Bi.size
    grid = math.ceil(n_samples / tpb)

    Bx = cupy.empty(Bi.size, dtype=Ax.dtype)

    kernel = _csr_sample_values_kern.get_function(
        _csr_sample_values_kern_types[(Ap.dtype, Ax.dtype)]
    )
    kernel((grid,), (tpb,),
           (n_row, n_col, Ap, Aj, Ax,
            n_samples, Bi, Bj, Bx))

    return Bx


_csr_sample_values_kern_types = _build_name_expressions(
    [(_int32_dtype, _float32_dtype), (_int32_dtype, _float64_dtype),
     (_int32_dtype, _complex64_dtype), (_int32_dtype, _complex128_dtype)],
    'csr_sample_values_kern')
_csr_sample_values_kern = core.RawModule(code="""
    #include <cupy/complex.cuh>
    template<typename I, typename T>
    __global__
    void csr_sample_values_kern(const I n_row,
                                const I n_col,
                                const I *Ap,
                                const I *Aj,
                                const T *Ax,
                                const I n_samples,
                                const I *Bi,
                                const I *Bj,
                                T *Bx) {

        // Get the index of the thread
        int n = blockIdx.x * blockDim.x + threadIdx.x;

        if(n < n_samples) {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // sample row
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // sample column

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            T x = 0;

            for(I jj = row_start; jj < row_end; jj++) {
                if (Aj[jj] == j)
                    x += Ax[jj];
            }

            Bx[n] = x;
        }
    }
""", options=_module_options, name_expressions=tuple(
    _csr_sample_values_kern_types.values()))


_csr_row_slice_kern_types = _build_name_expressions(
    [(_int32_dtype, _float32_dtype), (_int32_dtype, _float64_dtype),
     (_int32_dtype, _complex64_dtype), (_int32_dtype, _complex128_dtype)],
    'csr_row_slice_kern')
_csr_row_slice_kern = core.RawModule(code="""
    #include <cupy/complex.cuh>
    template<typename I, typename T>
    __global__
    void csr_row_slice_kern(const I start,
                            const I stop,
                            const I step,
                            const I *Ap,
                            const I *Aj,
                            const T *Ax,
                            const I *Bp,
                            I *Bj,
                            T *Bx) {

        // Get the index of the thread
        int out_row = blockIdx.x * blockDim.x + threadIdx.x;

        if (step > 0) {


            I in_row = out_row*step + start;
            if(in_row < stop) {

                I out_row_offset = Bp[out_row];

                const I row_start = Ap[in_row];
                const I row_end   = Ap[in_row+1];

                for(I jj = row_start; jj < row_end; jj++) {
                    Bj[out_row_offset] = Aj[jj];
                    Bx[out_row_offset] = Ax[jj];
                    out_row_offset++;
                }
            }

        } else {
            I in_row = out_row*step + start;
            if(in_row > stop) {

                I out_row_offset = Bp[out_row];

                const I row_start = Ap[in_row];
                const I row_end   = Ap[in_row+1];

                for(I jj = row_start; jj < row_end; jj++) {
                    Bj[out_row_offset] = Aj[jj];
                    Bx[out_row_offset] = Ax[jj];

                    out_row_offset++;
                }
            }
        }
    }
""", options=_module_options, name_expressions=tuple(
    _csr_row_slice_kern_types.values()))


def _csr_row_slice(start, stop, step, Ap, Aj, Ax, Bp, tpb=1024):
    """Populate indices and data arrays of sparse matrix by slicing the
    rows of an input sparse matrix

    Args
        start : starting row
        stop : ending row
        step : step increment size
        Ap : indptr array of input sparse matrix
        Aj : indices array of input sparse matrix
        Ax : data array of input sparse matrix
        Bp : indices array of output sparse matrix
        tpb : threads per block of row slice kernel

    Returns
        Bj : indices array of output sparse matrix
        Bx : data array of output sparse matrix
    """
    grid = math.ceil((len(Bp)-1) / tpb)

    nnz = Bp[-1].item()
    Bj = cupy.empty(nnz, dtype=Ap.dtype)
    Bx = cupy.empty(nnz, dtype=Ax.dtype)

    kernel = _csr_row_slice_kern.get_function(
        _csr_row_slice_kern_types[(Ap.dtype, Ax.dtype)]
    )
    kernel((grid,), (tpb,),
           (start, stop, step,
            Ap, Aj, Ax,
            Bp, Bj, Bx))

    return Bj, Bx


_check_idx_bounds_ker_types = _build_name_expressions(
    [_int32_dtype, _int64_dtype], 'idx_bounds_ker')
_check_idx_bounds_ker = core.RawModule(code="""
    template<typename I>
    __global__
    void idx_bounds_ker(const I *idxs,
                        const I idx_length,
                        const I bounds,
                        bool *results) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < idx_length) {
            const I cur_idx = idxs[i];

            if(cur_idx >= bounds)
                results[0] = true;

            if(cur_idx < -idx_length)
                results[1] = true;

            if(cur_idx < 0)
                results[2] = true;
        }
    }

""", options=_module_options, name_expressions=tuple(
    _check_idx_bounds_ker_types.values()))


def _check_bounds(idxs, bounds, tpb=1024):
    """Validate indices are within given bounds.
    """

    # This function doesn't use atomics, which would normally be used to
    # compute min/max. Further, only a single stream synchronization is
    # needed to check these bounds, rather than the 3 that would be required
    # if the 3 bounds checks were performed separately.

    grid = math.ceil(idxs.size / tpb)
    kernel = _check_idx_bounds_ker.get_function(
        _check_idx_bounds_ker_types[idxs.dtype]
    )
    results = cupy.array([False, False, False], dtype='bool')
    kernel((grid,), (tpb,),
           (idxs, idxs.size,
            bounds, results))

    # Only need to perform 1 host synchronization instead of 3
    res_host = results.get()

    return res_host[2], res_host[1], res_host[0]


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
        row, col = _broadcast_arrays(row, col)
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

        # Check bounds
        is_neg, is_out_lower, is_out_upper = _check_bounds(x, length)

        if is_out_upper:
            raise IndexError('index (%d) out of range' % x.max())

        if is_neg:
            if is_out_lower:
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
        x, _ = _broadcast_arrays(x, row)
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
