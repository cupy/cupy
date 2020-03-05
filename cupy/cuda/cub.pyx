# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

from cpython cimport sequence

import numpy

from cupy.core.core cimport ndarray, _internal_ascontiguousarray
from cupy.core.core cimport _internal_asfortranarray
from cupy.core.internal cimport _contig_axes
from cupy.cuda cimport memory
from cupy.cuda cimport stream
from cupy.cuda.driver cimport Stream as Stream_t

cimport cython


###############################################################################
# Const
###############################################################################

cdef enum:
    CUPY_CUB_INT8 = 0
    CUPY_CUB_UINT8 = 1
    CUPY_CUB_INT16 = 2
    CUPY_CUB_UINT16 = 3
    CUPY_CUB_INT32 = 4
    CUPY_CUB_UINT32 = 5
    CUPY_CUB_INT64 = 6
    CUPY_CUB_UINT64 = 7
    CUPY_CUB_FLOAT16 = 8
    CUPY_CUB_FLOAT32 = 9
    CUPY_CUB_FLOAT64 = 10
    CUPY_CUB_COMPLEX64 = 11
    CUPY_CUB_COMPLEX128 = 12

CUB_support_dtype = [numpy.int8, numpy.uint8,
                     numpy.int16, numpy.uint16,
                     numpy.int32, numpy.uint32,
                     numpy.int64, numpy.uint64,
                     numpy.float32, numpy.float64,
                     numpy.complex64, numpy.complex128]

###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cub.h' nogil:
    void cub_device_reduce(void*, size_t&, void*, void*, int, Stream_t,
                           int, int)
    void cub_device_segmented_reduce(void*, size_t&, void*, void*, int, void*,
                                     void*, Stream_t, int, int)
    void cub_device_spmv(void*, size_t&, void*, void*, void*, void*, void*,
                         int, int, int, Stream_t, int)
    size_t cub_device_reduce_get_workspace_size(void*, void*, int, Stream_t,
                                                int, int)
    size_t cub_device_segmented_reduce_get_workspace_size(
        void*, void*, int, void*, void*, Stream_t, int, int)
    size_t cub_device_spmv_get_workspace_size(
        void*, void*, void*, void*, void*, int, int, int, Stream_t, int)

###############################################################################
# Python interface
###############################################################################

cdef tuple _get_output_shape(ndarray arr, tuple out_axis, bint keepdims):
    cdef tuple out_shape

    if not keepdims:
        out_shape = tuple([arr.shape[axis] for axis in out_axis])
    else:
        out_shape = tuple([arr.shape[axis] if axis in out_axis else 1
                           for axis in range(arr.ndim)])
    return out_shape


cpdef Py_ssize_t _preprocess_array(ndarray arr, tuple reduce_axis,
                                   tuple out_axis, str order):
    '''
    This function more or less follows the logic of _get_permuted_args() in
    reduction.pxi. The input array arr is C- or F- contiguous along axis.
    '''
    cdef tuple axis_permutes, out_shape
    cdef Py_ssize_t contiguous_size = 1

    # one more sanity check?
    if order == 'C':
        axis_permutes = out_axis + reduce_axis
    elif order == 'F':
        axis_permutes = reduce_axis + out_axis
    assert axis_permutes == tuple(range(len(arr.shape)))

    for axis in reduce_axis:
        contiguous_size *= arr.shape[axis]
    return contiguous_size


def device_reduce(ndarray x, op, tuple out_axis, out=None,
                  bint keepdims=False):
    cdef ndarray y
    cdef memory.MemoryPointer ws
    cdef int dtype_id, ndim_out, kv_bytes, x_size, _op
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *y_ptr
    cdef void *ws_ptr
    cdef Stream_t s
    cdef tuple out_shape

    if keepdims:
        out_shape = _get_output_shape(x, out_axis, keepdims)
        ndim_out = len(out_shape)
    else:
        ndim_out = 0

    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            'output parameter for reduction operation has the wrong number of '
            'dimensions')
    if op < CUPY_CUB_SUM or op > CUPY_CUB_ARGMAX:
        raise ValueError('only CUPY_CUB_SUM, CUPY_CUB_MIN, CUPY_CUB_MAX, '
                         'CUPY_CUB_ARGMIN, and CUPY_CUB_ARGMAX are supported.')
    if x.size == 0 and op != CUPY_CUB_SUM:
        raise ValueError('zero-size array to reduction operation {} which has '
                         'no identity'.format(op.name))
    x = _internal_ascontiguousarray(x)

    if CUPY_CUB_SUM <= op <= CUPY_CUB_MAX:
        y = ndarray((), x.dtype)
    else:  # argmin and argmax
        # cub::KeyValuePair has 1 int + 1 arbitrary type
        kv_bytes = (4 + x.dtype.itemsize)
        y = ndarray((kv_bytes,), numpy.int8)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = _get_dtype_id(x.dtype)
    s = <Stream_t>stream.get_current_stream_ptr()
    x_size = <int>x.size
    ws_size = cub_device_reduce_get_workspace_size(x_ptr, y_ptr, x.size, s,
                                                   op, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void *>ws.ptr
    _op = <int>op
    with nogil:
        cub_device_reduce(ws_ptr, ws_size, x_ptr, y_ptr, x_size, s, _op,
                          dtype_id)
    if op == CUPY_CUB_ARGMIN or op == CUPY_CUB_ARGMAX:
        # get key from KeyValuePair: need to reinterpret the first 4 bytes
        # and then cast it
        y = y[0:4].view(numpy.int32).astype(numpy.int64)[0]
        y = y.reshape(())

    if keepdims:
        y = y.reshape(out_shape)
    if out is not None:
        out[...] = y
        y = out
    return y


def device_segmented_reduce(ndarray x, op, tuple reduce_axis,
                            tuple out_axis, out=None, bint keepdims=False):
    # if import at the top level, a segfault would happen when import cupy!
    from cupy.creation.ranges import arange

    cdef ndarray y, offset
    cdef str order
    cdef memory.MemoryPointer ws
    cdef void* x_ptr
    cdef void* y_ptr
    cdef void* ws_ptr
    cdef void* offset_start_ptr
    cdef int dtype_id, n_segments, _op
    cdef size_t ws_size
    cdef Py_ssize_t contiguous_size
    cdef tuple out_shape
    cdef Stream_t s

    if op < CUPY_CUB_SUM or op > CUPY_CUB_MAX:
        raise ValueError('only CUPY_CUB_SUM, CUPY_CUB_MIN, and CUPY_CUB_MAX '
                         'are supported.')
    if x.size == 0 and op != CUPY_CUB_SUM:
        raise ValueError('zero-size array to reduction operation {} which has '
                         'no identity'.format(op.name))
    if x.flags.c_contiguous:
        order = 'C'
    elif x.flags.f_contiguous:
        order = 'F'
    else:  # impossible at this point, just in case
        raise RuntimeError('input is neither C- nor F- contiguous.')

    # prepare input
    contiguous_size = _preprocess_array(x, reduce_axis, out_axis, order)
    out_shape = _get_output_shape(x, out_axis, keepdims)
    x_ptr = <void*>x.data.ptr
    y = ndarray(out_shape, dtype=x.dtype, order=order)
    y_ptr = <void*>y.data.ptr
    if out is not None and out.shape != out_shape:
        raise ValueError(
            'output parameter for reduction operation has the wrong shape')
    if x.size == 0:  # for CUPY_CUB_SUM
        if out is not None:
            y = out
        y[...] = 0
        return y
    n_segments = x.size//contiguous_size
    # CUB internally use int for offset...
    offset = arange(0, x.size+1, contiguous_size, dtype=numpy.int32)
    offset_start_ptr = <void*>offset.data.ptr
    offset_end_ptr = <void*>((<int*><void*>offset.data.ptr)+1)
    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = _get_dtype_id(x.dtype)

    # get workspace size and then fire up
    ws_size = cub_device_segmented_reduce_get_workspace_size(
        x_ptr, y_ptr, n_segments, offset_start_ptr, offset_end_ptr, s,
        op, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void*>ws.ptr
    _op = <int>op
    with nogil:
        cub_device_segmented_reduce(ws_ptr, ws_size, x_ptr, y_ptr, n_segments,
                                    offset_start_ptr, offset_end_ptr, s,
                                    _op, dtype_id)

    if out is not None:
        out[...] = y
        y = out
    return y


def device_csrmv(int n_rows, int n_cols, int nnz, ndarray values,
                 ndarray indptr, ndarray indices, ndarray x):
    cdef ndarray y
    cdef memory.MemoryPointer ws
    cdef void* values_ptr
    cdef void* row_offsets_ptr
    cdef void* col_indices_ptr
    cdef void* x_ptr
    cdef void* y_ptr
    cdef void* ws_ptr
    cdef int dtype_id
    cdef size_t ws_size
    cdef Stream_t s

    if x.ndim != 1:
        raise ValueError('array must be 1d')
    if x.size != n_cols:
        raise ValueError("size of array does not match the CSR matrix")

    if values.dtype == x.dtype:
        dtype = values.dtype
    else:
        dtype = numpy.promote_types(values.dtype, x.dtype)
        values = values.astype(dtype, "C", None, None, False)
        x = x.astype(dtype, "C", None, None, False)

    # CSR matrix attributes
    values_ptr = <void*>values.data.ptr
    row_offsets_ptr = <void*>indptr.data.ptr
    col_indices_ptr = <void*>indices.data.ptr

    x_ptr = <void*>x.data.ptr

    # prepare output array
    y = ndarray((n_rows,), dtype=dtype)
    y_ptr = <void*>y.data.ptr

    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = _get_dtype_id(dtype)

    # get workspace size and then fire up
    ws_size = cub_device_spmv_get_workspace_size(
        values_ptr, row_offsets_ptr, col_indices_ptr, x_ptr, y_ptr, n_rows,
        n_cols, nnz, s, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void *>ws.ptr
    with nogil:
        cub_device_spmv(ws_ptr, ws_size, values_ptr, row_offsets_ptr,
                        col_indices_ptr, x_ptr, y_ptr, n_rows, n_cols, nnz, s,
                        dtype_id)

    return y


cdef bint _cub_device_segmented_reduce_axis_compatible(
        tuple cub_axis, Py_ssize_t ndim, order):
    # Implementation borrowed from cupy.fft.fft._get_cufft_plan_nd().
    # This function checks if the reduced axes are contiguous.

    # the axes to be reduced must be C- or F- contiguous
    if _contig_axes(cub_axis):
        if order in ('c', 'C'):
            return ((ndim - 1) in cub_axis)
        elif order in ('f', 'F'):
            return (0 in cub_axis)
    return False


def can_use_device_reduce(int op, x_dtype, tuple out_axis, dtype=None):
    return out_axis is () and _cub_reduce_dtype_compatible(x_dtype, op, dtype)


def can_use_device_segmented_reduce(int op, x_dtype, Py_ssize_t ndim,
                                    reduce_axis, dtype=None, order='C'):
    if not _cub_reduce_dtype_compatible(x_dtype, op, dtype):
        return False
    return _cub_device_segmented_reduce_axis_compatible(reduce_axis, ndim,
                                                        order)


cdef _cub_reduce_dtype_compatible(x_dtype, int op, dtype=None,
                                  bint segmented=False):
    if dtype is None:
        if op == CUPY_CUB_SUM:
            # auto dtype:
            # CUB reduce_sum does not support dtype promotion.
            # See _sum_auto_dtype in cupy/core/_routines_math.pyx for which
            # dtypes are promoted.
            support_dtype = [numpy.int64, numpy.uint64,
                             numpy.float32, numpy.float64,
                             numpy.complex64, numpy.complex128]
        else:
            support_dtype = CUB_support_dtype
    elif dtype == x_dtype:
        support_dtype = CUB_support_dtype
    else:
        return False
    if x_dtype not in support_dtype:
        return False
    return True


def cub_reduction(arr, op, axis=None, dtype=None, out=None, keepdims=False):
    """Perform a reduction using CUB.

    If the specified reduction is not possible, None is returned.
    """
    # if import at the top level, a segfault would happen when import cupy!
    from cupy.core._kernel import _get_axis
    cdef bint enforce_numpy_API = False

    if op > CUPY_CUB_MAX:
        # For argmin and argmax, NumPy does not allow a tuple for axis.
        # Also, the keepdims and dtype kwargs are not provided.
        #
        # For now we don't enforce these for consistency with existing CuPy
        # non-CUB reduction behavior.
        # https://github.com/cupy/cupy/issues/2595
        enforce_numpy_API = False
        if enforce_numpy_API:
            # numpy's argmin and argmax do not support a tuple of axes
            if sequence.PySequence_Check(axis):
                raise TypeError(
                    "'tuple' object cannot be interpreted as an integer")
            if keepdims:
                raise TypeError(
                    "'keepdims' is an invalid keyword argument for "
                    "argmin or argmax.")
            if dtype is not None:
                raise TypeError(
                    "'dtype' is an invalid keyword argument for "
                    "argmin or argmax.")
        else:
            if dtype is not None:
                # fallback to existing non-CUB behavior
                return None

    reduce_axis, out_axis = _get_axis(axis, arr.ndim)
    if can_use_device_reduce(op, arr.dtype, out_axis, dtype):
        return device_reduce(arr, op, out_axis, out, keepdims)

    if op > CUPY_CUB_MAX:
        # segmented reduction not currently implemented for argmax, argmin
        return None

    if arr.flags.c_contiguous:
        order = 'C'
    elif arr.flags.f_contiguous:
        order = 'F'
    else:
        order = None

    if can_use_device_segmented_reduce(op, arr.dtype, arr.ndim,
                                       reduce_axis, dtype, order):
        return device_segmented_reduce(arr, op, reduce_axis, out_axis,
                                       out, keepdims)
    return None


def _get_dtype_id(dtype):
    if dtype == numpy.int8:
        ret = CUPY_CUB_INT8
    elif dtype == numpy.uint8:
        ret = CUPY_CUB_UINT8
    elif dtype == numpy.int16:
        ret = CUPY_CUB_INT16
    elif dtype == numpy.uint16:
        ret = CUPY_CUB_UINT16
    elif dtype == numpy.int32:
        ret = CUPY_CUB_INT32
    elif dtype == numpy.uint32:
        ret = CUPY_CUB_UINT32
    elif dtype == numpy.int64:
        ret = CUPY_CUB_INT64
    elif dtype == numpy.uint64:
        ret = CUPY_CUB_UINT64
    elif dtype == numpy.float32:
        ret = CUPY_CUB_FLOAT32
    elif dtype == numpy.float64:
        ret = CUPY_CUB_FLOAT64
    elif dtype == numpy.complex64:
        ret = CUPY_CUB_COMPLEX64
    elif dtype == numpy.complex128:
        ret = CUPY_CUB_COMPLEX128
    else:
        raise ValueError('Unsupported dtype ({})'.format(dtype))
    return ret
