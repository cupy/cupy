# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

from cpython cimport sequence
from libc.stdint cimport intptr_t

from cupy_backends.cuda.api cimport runtime
from cupy._core.core cimport _internal_ascontiguousarray
from cupy._core.internal cimport _contig_axes, is_in
from cupy.cuda cimport common
from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy.cuda cimport stream

import cupy
import cupy._core as _core
from cupy.cuda import memory as _memory
import numpy


###############################################################################
# Const
###############################################################################

CUB_support_dtype_without_half = [numpy.int8, numpy.uint8,
                                  numpy.int16, numpy.uint16,
                                  numpy.int32, numpy.uint32,
                                  numpy.int64, numpy.uint64,
                                  numpy.float32, numpy.float64,
                                  numpy.complex64, numpy.complex128]

CUB_support_dtype_with_half = CUB_support_dtype_without_half + [numpy.float16]

CUB_support_dtype = {}

CUB_sum_support_dtype_without_half = [numpy.int64, numpy.uint64,
                                      numpy.float32, numpy.float64,
                                      numpy.complex64, numpy.complex128]

CUB_sum_support_dtype_with_half = \
    CUB_sum_support_dtype_without_half + [numpy.float16]

CUB_sum_support_dtype = {}


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cub.h' nogil:
    ctypedef void* Stream_t 'cudaStream_t'

    void cub_device_reduce(void*, size_t&, void*, void*, int, Stream_t,
                           int, int)
    void cub_device_segmented_reduce(void*, size_t&, void*, void*, int, int,
                                     Stream_t, int, int)
    void cub_device_spmv(void*, size_t&, void*, void*, void*, void*, void*,
                         int, int, int, Stream_t, int)
    void cub_device_scan(void*, size_t&, void*, void*, int, Stream_t, int, int)
    void cub_device_histogram_range(void*, size_t&, void*, void*, int, void*,
                                    size_t, Stream_t, int)
    void cub_device_histogram_even(void*, size_t&, void*, void*, int, int, int,
                                   size_t, Stream_t, int)
    size_t cub_device_reduce_get_workspace_size(void*, void*, int, Stream_t,
                                                int, int)
    size_t cub_device_segmented_reduce_get_workspace_size(
        void*, void*, int, int, Stream_t, int, int)
    size_t cub_device_spmv_get_workspace_size(
        void*, void*, void*, void*, void*, int, int, int, Stream_t, int)
    size_t cub_device_scan_get_workspace_size(
        void*, void*, int, Stream_t, int, int)
    size_t cub_device_histogram_range_get_workspace_size(
        void*, void*, int, void*, size_t, Stream_t, int)
    size_t cub_device_histogram_even_get_workspace_size(
        void*, void*, int, int, int, size_t, Stream_t, int)

    # Build-time version
    int CUPY_CUB_VERSION_CODE


###############################################################################
# Python interface
###############################################################################

available = True


def get_build_version():
    if CUPY_CUB_VERSION_CODE == -1:
        return '<unknown>'
    return CUPY_CUB_VERSION_CODE


cpdef _get_cuda_build_version():
    if CUPY_CUDA_VERSION > 0:
        return CUPY_CUDA_VERSION
    elif CUPY_HIP_VERSION > 0:
        return CUPY_HIP_VERSION
    else:
        return 0


cdef tuple _get_output_shape(_ndarray_base arr, tuple out_axis, bint keepdims):
    cdef tuple out_shape

    if not keepdims:
        out_shape = tuple([arr.shape[axis] for axis in out_axis])
    else:
        out_shape = tuple([arr.shape[axis] if axis in out_axis else 1
                           for axis in range(arr.ndim)])
    return out_shape


cpdef Py_ssize_t _preprocess_array(tuple arr_shape, tuple reduce_axis,
                                   tuple out_axis, str order) except -1:
    '''
    This function more or less follows the logic of _get_permuted_args() in
    reduction.pxi. The input array arr is C- or F- contiguous along axis.
    '''
    cdef tuple axis_permutes
    cdef Py_ssize_t contiguous_size = 1

    # one more sanity check?
    if order == 'C':
        axis_permutes = out_axis + reduce_axis
    elif order == 'F':
        axis_permutes = reduce_axis + out_axis
    elif order == 'CF':
        axis_permutes = tuple(set(out_axis + reduce_axis))
    assert axis_permutes == tuple(range(len(arr_shape)))

    for axis in reduce_axis:
        contiguous_size *= arr_shape[axis]
    return contiguous_size


def device_reduce(_ndarray_base x, op, tuple reduce_axis, tuple out_axis,
                  out=None, bint keepdims=False):
    cdef _ndarray_base y
    cdef memory.MemoryPointer ws
    cdef int dtype_id, ndim_out, kv_bytes, x_size, op_code
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
    reduce_shape = _get_output_shape(x, reduce_axis, False)

    if out is not None and out.ndim != ndim_out:
        raise ValueError(
            'output parameter for reduction operation has the wrong number of '
            'dimensions')
    if op not in (CUPY_CUB_SUM, CUPY_CUB_PROD, CUPY_CUB_MIN, CUPY_CUB_MAX,
                  CUPY_CUB_ARGMIN, CUPY_CUB_ARGMAX):
        raise ValueError('only CUPY_CUB_SUM, CUPY_CUB_PROD, CUPY_CUB_MIN, '
                         'CUPY_CUB_MAX, CUPY_CUB_ARGMIN, and CUPY_CUB_ARGMAX '
                         'are supported.')
    if op not in (CUPY_CUB_SUM, CUPY_CUB_PROD) and is_in(reduce_shape, 0):
        raise ValueError('zero-size array to reduction operation {} which has '
                         'no identity'.format(op.name))
    x = _internal_ascontiguousarray(x)

    if op in (CUPY_CUB_SUM, CUPY_CUB_PROD, CUPY_CUB_MIN, CUPY_CUB_MAX):
        y = _core.ndarray((), x.dtype)
    else:  # argmin and argmax
        # cub::KeyValuePair has 1 int + 1 arbitrary type
        # Note that:
        # - The key may be padded to make the value aligned.
        # - Unlike a regular structure, thrust::complex<T> is aligned to the
        #   twice of the size of T.
        if x.dtype.char in 'FD':
            kv_bytes = x.dtype.alignment * 2 + x.dtype.itemsize
        else:
            kv_bytes = max(4, x.dtype.alignment) + x.dtype.itemsize
        y = _core.ndarray((kv_bytes,), numpy.int8)
    x_ptr = <void *>x.data.ptr
    y_ptr = <void *>y.data.ptr
    dtype_id = common._get_dtype_id(x.dtype)
    s = <Stream_t>stream.get_current_stream_ptr()
    x_size = <int>x.size
    ws_size = cub_device_reduce_get_workspace_size(x_ptr, y_ptr, x.size, s,
                                                   op, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void *>ws.ptr
    op_code = <int>op
    with nogil:
        cub_device_reduce(ws_ptr, ws_size, x_ptr, y_ptr, x_size, s, op_code,
                          dtype_id)
    if op in (CUPY_CUB_ARGMIN, CUPY_CUB_ARGMAX):
        # get key from KeyValuePair: need to reinterpret the first 4 bytes
        # and then cast it
        y = y[0:4].view(numpy.int32).astype(numpy.int64)[0]
        y = y.reshape(())

    if keepdims:
        y = y.reshape(out_shape)
    if out is not None:
        cupy._core.elementwise_copy(y, out)
        y = out
    return y


def device_segmented_reduce(_ndarray_base x, op, tuple reduce_axis,
                            tuple out_axis, out=None, bint keepdims=False,
                            Py_ssize_t contiguous_size=0):
    cdef _ndarray_base y
    cdef str order
    cdef memory.MemoryPointer ws
    cdef void* x_ptr
    cdef void* y_ptr
    cdef void* ws_ptr
    cdef int dtype_id, n_segments, op_code
    cdef size_t ws_size
    cdef tuple out_shape
    cdef Stream_t s

    if op not in (CUPY_CUB_SUM, CUPY_CUB_PROD, CUPY_CUB_MIN, CUPY_CUB_MAX):
        raise ValueError('only CUPY_CUB_SUM, CUPY_CUB_PROD, CUPY_CUB_MIN, '
                         'and CUPY_CUB_MAX are supported.')
    reduce_shape = _get_output_shape(x, reduce_axis, False)
    if op not in (CUPY_CUB_SUM, CUPY_CUB_PROD) and is_in(reduce_shape, 0):
        raise ValueError('zero-size array to reduction operation {} which has '
                         'no identity'.format(op.name))
    if x.flags.c_contiguous:
        order = 'C'
    elif x.flags.f_contiguous:
        order = 'F'
    else:  # impossible at this point, just in case
        raise RuntimeError('input is neither C- nor F- contiguous.')

    # prepare input
    out_shape = _get_output_shape(x, out_axis, keepdims)
    x_ptr = <void*>x.data.ptr
    y = _core.ndarray(out_shape, dtype=x.dtype, order=order)
    y_ptr = <void*>y.data.ptr
    if out is not None and out.shape != out_shape:
        raise ValueError(
            'output parameter for reduction operation has the wrong shape')
    if x.size == 0:  # for CUPY_CUB_SUM & CUPY_CUB_PROD
        if out is not None:
            y = out
        if op == CUPY_CUB_SUM:
            y.fill(0)
        elif op == CUPY_CUB_PROD:
            y.fill(1)
        return y
    n_segments = x.size//contiguous_size
    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = common._get_dtype_id(x.dtype)

    # get workspace size and then fire up
    ws_size = cub_device_segmented_reduce_get_workspace_size(
        x_ptr, y_ptr, n_segments, contiguous_size, s, op, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void*>ws.ptr
    op_code = <int>op
    with nogil:
        cub_device_segmented_reduce(ws_ptr, ws_size, x_ptr, y_ptr, n_segments,
                                    contiguous_size, s, op_code, dtype_id)

    if out is not None:
        cupy._core.elementwise_copy(y, out)
        y = out
    return y


def device_csrmv(int n_rows, int n_cols, int nnz, _ndarray_base values,
                 _ndarray_base indptr, _ndarray_base indices, _ndarray_base x):
    cdef _ndarray_base y
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
    if runtime._is_hip_environment:
        raise RuntimeError("hipCUB does not support SpMV")

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
    y = _core.ndarray((n_rows,), dtype=dtype)
    y_ptr = <void*>y.data.ptr

    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = common._get_dtype_id(dtype)

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


def device_scan(_ndarray_base x, op):
    cdef memory.MemoryPointer ws
    cdef int dtype_id, x_size, op_code
    cdef size_t ws_size
    cdef void *x_ptr
    cdef void *ws_ptr
    cdef Stream_t s

    if op not in (CUPY_CUB_CUMSUM, CUPY_CUB_CUMPROD):
        raise ValueError('only CUPY_CUB_CUMSUM and CUPY_CUB_CUMPROD '
                         'are supported.')

    # determine shape: x is either 1D (with axis=None,0) or ND but ravelled.
    x_size = <int>x.size
    if x_size == 0:
        return x

    x = _internal_ascontiguousarray(x)
    x_ptr = <void *>x.data.ptr
    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = common._get_dtype_id(x.dtype)
    ws_size = cub_device_scan_get_workspace_size(x_ptr, x_ptr, x_size, s,
                                                 op, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void *>ws.ptr
    op_code = <int>op
    with nogil:
        # the scan is in-place
        cub_device_scan(ws_ptr, ws_size, x_ptr, x_ptr, x_size, s,
                        op_code, dtype_id)
    return x


def device_histogram(_ndarray_base x, _ndarray_base y, bins):
    cdef memory.MemoryPointer ws
    cdef size_t ws_size, n_samples
    cdef int dtype_id, n_bins
    cdef void* x_ptr
    cdef void* bins_ptr
    cdef void* y_ptr
    cdef void* ws_ptr
    cdef Stream_t s
    cdef bint is_even

    # TODO(leofang): perhaps not needed?
    # y is guaranteed contiguous
    x = _internal_ascontiguousarray(x)
    if isinstance(bins, _ndarray_base):
        bins = _internal_ascontiguousarray(bins)
        bins_ptr = <void*><intptr_t>bins.data.ptr
        n_bins = bins.size
        is_even = False
    else:
        n_bins = bins
        is_even = True
        if runtime._is_hip_environment:
            raise RuntimeError("not supported yet")
        if x.dtype.kind not in 'bui':
            raise ValueError("only integer input is supported")
    assert y.size == n_bins - 1

    x_ptr = <void*>x.data.ptr
    y_ptr = <void*>y.data.ptr
    n_samples = x.size
    s = <Stream_t>stream.get_current_stream_ptr()
    dtype_id = common._get_dtype_id(x.dtype)

    if is_even:
        ws_size = cub_device_histogram_even_get_workspace_size(
            x_ptr, y_ptr, n_bins, 0, n_bins-1, n_samples, s, dtype_id)
    else:
        ws_size = cub_device_histogram_range_get_workspace_size(
            x_ptr, y_ptr, n_bins, bins_ptr, n_samples, s, dtype_id)
    ws = memory.alloc(ws_size)
    ws_ptr = <void*>ws.ptr

    with nogil:
        if is_even:
            cub_device_histogram_even(
                ws_ptr, ws_size, x_ptr, y_ptr, n_bins, 0, n_bins-1,
                n_samples, s, dtype_id)
        else:
            cub_device_histogram_range(
                ws_ptr, ws_size, x_ptr, y_ptr, n_bins,
                bins_ptr, n_samples, s, dtype_id)
    return y


cpdef bint _cub_device_segmented_reduce_axis_compatible(
        tuple cub_axis, Py_ssize_t ndim, str order):
    # This function checks if the reduced axes are C- or F- contiguous.
    if _contig_axes(cub_axis):
        if order == 'C':
            return (cub_axis[-1] == (ndim - 1))
        elif order == 'F':
            return (cub_axis[0] == 0)
    return False


cdef bint can_use_device_reduce(
        _ndarray_base x, int op, tuple out_axis, dtype=None) except*:
    return (
        out_axis is ()
        and _cub_reduce_dtype_compatible(x.dtype, op, dtype)
        and x.size <= 0x7fffffff)  # until we resolve cupy/cupy#3309


cdef (bint, Py_ssize_t) can_use_device_segmented_reduce(  # noqa: E211
        _ndarray_base x, int op, tuple reduce_axis, tuple out_axis,
        dtype=None, str order='C') except*:
    if not _cub_reduce_dtype_compatible(x.dtype, op, dtype):
        return (False, 0)
    # NumPy/CuPy quirk: zero-size arrays are both C- & F- contig...
    if x.size != 0:
        if not _cub_device_segmented_reduce_axis_compatible(
                reduce_axis, x.ndim, order):
            return (False, 0)
    else:
        order = 'CF'  # for computing the contig size
    # until we resolve cupy/cupy#3309
    cdef Py_ssize_t contiguous_size = _preprocess_array(
        x.shape, reduce_axis, out_axis, order)
    return (contiguous_size <= 0x7fffffff, contiguous_size)


cdef _cub_support_dtype(bint sum_mode, int dev_id):
    if sum_mode:
        support_dtype_dict = CUB_sum_support_dtype
        with_half = CUB_sum_support_dtype_with_half
        without_half = CUB_sum_support_dtype_without_half
    else:
        support_dtype_dict = CUB_support_dtype
        with_half = CUB_support_dtype_with_half
        without_half = CUB_support_dtype_without_half

    if dev_id not in support_dtype_dict:
        if common._is_fp16_supported():
            support_dtype = with_half
        else:
            support_dtype = without_half

        support_dtype_dict[dev_id] = support_dtype

    return support_dtype_dict[dev_id]


cdef _cub_reduce_dtype_compatible(x_dtype, int op, dtype=None):
    cdef int dev_id = device.get_device_id()

    if dtype is None:
        if op in (CUPY_CUB_SUM, CUPY_CUB_PROD):
            # auto dtype:
            # CUB reduce_sum does not support dtype promotion.
            # See _sum_auto_dtype in cupy/_core/_routines_math.pyx for which
            # dtypes are promoted.
            support_dtype = _cub_support_dtype(True, dev_id)
        else:
            support_dtype = _cub_support_dtype(False, dev_id)
    elif dtype == x_dtype:
        support_dtype = _cub_support_dtype(False, dev_id)
    else:
        return False

    if x_dtype not in support_dtype:
        return False
    return True


cpdef cub_reduction(
        _ndarray_base arr, op,
        axis=None, dtype=None, _ndarray_base out=None, keepdims=False):
    """Perform a reduction using CUB.

    If the specified reduction is not possible, None is returned.
    """
    # if import at the top level, a segfault would happen when import cupy!
    from cupy._core._reduction import _get_axis
    cdef bint enforce_numpy_API = False, is_ok
    cdef str order
    cdef tuple reduce_axis, out_axis
    cdef Py_ssize_t contiguous_size

    if op in (CUPY_CUB_ARGMIN, CUPY_CUB_ARGMAX):
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

    if arr._c_contiguous:
        order = 'C'
    elif arr._f_contiguous:
        order = 'F'
    else:
        return None

    reduce_axis, out_axis = _get_axis(axis, arr.ndim)
    if can_use_device_reduce(arr, op, out_axis, dtype):
        return device_reduce(arr, op, reduce_axis, out_axis, out, keepdims)

    if op in (CUPY_CUB_ARGMIN, CUPY_CUB_ARGMAX):
        # segmented reduction not currently implemented for argmax, argmin
        return None

    is_ok, contiguous_size = can_use_device_segmented_reduce(
        arr, op, reduce_axis, out_axis, dtype, order)
    if is_ok and contiguous_size > 0:
        return device_segmented_reduce(arr, op, reduce_axis, out_axis,
                                       out, keepdims, contiguous_size)
    return None


cpdef cub_scan(_ndarray_base arr, op):
    """Perform an (in-place) prefix scan using CUB.

    If the specified scan is not possible, None is returned.
    """
    if op not in (CUPY_CUB_CUMSUM, CUPY_CUB_CUMPROD):
        return None

    x_dtype = arr.dtype
    if x_dtype == numpy.complex128:
        # cub_device_scan seems buggy for complex128:
        # https://github.com/cupy/cupy/pull/2919#issuecomment-574633590
        return None

    cdef int dev_id = device.get_device_id()
    if x_dtype in _cub_support_dtype(False, dev_id):
        return device_scan(arr, op)

    return None


cpdef cub_histogram(_ndarray_base x, _ndarray_base y, bins):
    """Check if the required workspace size is too much, if not then proceed
    to compute the histogram, otherwise return None. This is a workaround for
    NVIDIA/cub#613.
    """
    try:
        out = device_histogram(x, y, bins)
    except _memory.OutOfMemoryError:
        return None
    return out
