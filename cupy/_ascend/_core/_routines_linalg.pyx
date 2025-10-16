import math
import os
import warnings

import cython
import numpy

import cupy
from cupy._core._kernel import ElementwiseKernel
#from cupy._core._reduction import ReductionKernel
from cupy._core._ufuncs import elementwise_copy
import cupy._core.core as core


from libc.stdint cimport intptr_t

from cupy._core cimport _accelerator
from cupy._core._carray cimport shape_t
from cupy._core._dtype cimport to_cuda_dtype
from cupy._core._scalar cimport get_typename
from cupy._core._routines_creation cimport _internal_ascontiguousarray
from cupy._core._routines_creation cimport _ndarray_init
from cupy._core._routines_creation cimport ascontiguousarray
from cupy._core.core cimport _ndarray_base
from cupy._core cimport _memory_range
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core cimport _routines_math as _math # use only multiply
from cupy.xpu cimport device
from cupy_backends.cuda.api cimport runtime


cdef extern from '../../../cupy_backends/cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y


cdef list compute_types = [COMPUTE_TYPE_TBD,  # float16
                           COMPUTE_TYPE_TBD,  # float32
                           COMPUTE_TYPE_TBD]  # float64
cdef dict compute_type_str = {
    0: 'COMPUTE_TYPE_TBD',
    1: 'COMPUTE_TYPE_DEFAULT',
    2: 'COMPUTE_TYPE_PEDANTIC',
    3: 'COMPUTE_TYPE_FP16',
    4: 'COMPUTE_TYPE_FP32',
    5: 'COMPUTE_TYPE_FP64',
    6: 'COMPUTE_TYPE_BF16',
    7: 'COMPUTE_TYPE_TF32',
}

# TODO ComputeType maybe diff for diff xpu backend
cpdef int to_compute_type_index(dtype) except -1:
    cdef str dtype_char = numpy.dtype(dtype).char
    if dtype_char == 'e':
        return 0
    elif dtype_char in 'fF':
        return 1
    elif dtype_char in 'dD':
        return 2
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))

cpdef get_compute_type(dtype):
    global compute_types
    cdef int index = to_compute_type_index(dtype)
    if compute_types[index] == COMPUTE_TYPE_TBD:
        compute_type = COMPUTE_TYPE_DEFAULT
        dtype_char = numpy.dtype(dtype).char
        if dtype_char in 'fF' and int(os.getenv('CUPY_TF32', '0')) > 0:
            compute_type = COMPUTE_TYPE_TF32
        set_compute_type(dtype, compute_type)
    return compute_types[index]

cpdef set_compute_type(dtype, compute_type):
    global compute_types
    if compute_type in (COMPUTE_TYPE_TBD, COMPUTE_TYPE_DEFAULT,
                        COMPUTE_TYPE_PEDANTIC, COMPUTE_TYPE_FP16,
                        COMPUTE_TYPE_FP32, COMPUTE_TYPE_FP64):
        compute_types[to_compute_type_index(dtype)] = compute_type
    elif compute_type in (COMPUTE_TYPE_BF16, COMPUTE_TYPE_TF32):
        if int(device.get_compute_capability()) >= 80:
            compute_types[to_compute_type_index(dtype)] = compute_type
        else:
            warnings.warn('COMPUTE_TYPE_BF16 and COMPUTE_TYPE_TF32 are only '
                          'available on GPUs with compute capability 8.0 or '
                          'higher. COMPUTE_TYPE_DEFAULT will be used instead.')
            compute_types[to_compute_type_index(dtype)] = COMPUTE_TYPE_DEFAULT
    else:
        raise ValueError('Unknown compute type: {}'.format(compute_type))


cpdef compute_type_to_str(compute_type):
    if compute_type in compute_type_str:
        return compute_type_str[compute_type]
    else:
        return compute_type

cdef _ascend_matmul(_ndarray_base a, _ndarray_base b, _ndarray_base out=None):
    # TODO launch_acl_func("ascend_matmul")
    pass

cpdef _ndarray_base matmul(
        _ndarray_base a, _ndarray_base b, _ndarray_base out=None):
    """Matrix product of two arrays.

    Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    Args:
        a (cupy.ndarray): The left argument.
        b (cupy.ndarray): The right argument.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.matmul`

    """
    #from cupy_backends.cuda.libs import cublas

    cdef Py_ssize_t i, n, m, ka, kb, a_sh, b_sh, c_sh, ldc
    cdef Py_ssize_t batchCount, a_part_outshape, b_part_outshape
    cdef int orig_a_ndim, orig_b_ndim, a_ndim, b_ndim, ndim
    cdef _ndarray_base ap, bp, cp, c_view
    cdef bint use_broadcast

    orig_a_ndim = a._shape.size()
    orig_b_ndim = b._shape.size()
    if orig_a_ndim == 0 or orig_b_ndim == 0:
        raise ValueError('Scalar operands are not allowed, use \'*\' instead')

    ndim = max(orig_a_ndim, orig_b_ndim)
    if ndim <= 2:
        IF CUPY_CANN_VERSION <= 0:
            if out is None:
                return dot(a, b, out)
            ret_dtype = numpy.promote_types(a.dtype, b.dtype)
            if out._c_contiguous and ret_dtype == out.dtype:
                return dot(a, b, out)
            c = _ndarray_init(cupy.ndarray, out._shape, dtype=ret_dtype, obj=None)
            dot(a, b, c)
            elementwise_copy(c, out)
            return out
        ELSE:
            out = _ascend_matmul(a, b, out)
            return out

    orig_a = a
    orig_b = b
    a_part_outshape = b_part_outshape = 0
    if orig_a_ndim == 1:
        a = _manipulation._reshape(a, (1, a.size))
    else:
        a = a.view()
        a_part_outshape = a._shape[orig_a_ndim - 2]
    if orig_b_ndim == 1:
        b = _manipulation._reshape(b, (b.size, 1))
        ldc = 1
    else:
        b = b.view()
        b_part_outshape = ldc = b._shape[orig_b_ndim - 1]

    # expand dims
    a_ndim = a._shape.size()
    b_ndim = b._shape.size()
    if a_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        a._set_shape_and_strides(
            (1,) * (ndim - a_ndim) + a.shape,
            (0,) * (ndim - a_ndim) + a.strides,
            True, True)
    if b_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        b._set_shape_and_strides(
            (1,) * (ndim - b_ndim) + b.shape,
            (0,) * (ndim - b_ndim) + b.strides,
            True, True)

    ret_dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = ret_dtype
    if dtype.char == 'e':
        dtype = numpy.dtype('f')

    a = ascontiguousarray(a, dtype)
    b = ascontiguousarray(b, dtype)

    # broadcast
    batchCount = 1  # batchCount = numpy.prod(out_shape[:-2])
    out_shape = []
    use_broadcast = False
    for i in range(0, ndim - 2):
        a_sh = a._shape[i]
        b_sh = b._shape[i]
        if a_sh != b_sh and a_sh != 1 and b_sh != 1:
            raise ValueError(
                'operands could not be broadcast together with '
                'remapped shapes')

        if a_sh == 0 or b_sh == 0:
            c_sh = 0
        else:
            c_sh = max(a_sh, b_sh)
        batchCount *= c_sh
        out_shape.append(c_sh)
        if a_sh == 1 and c_sh > 1:
            a._strides[i] = 0
            a._shape[i] = c_sh
            a._c_contiguous = a._f_contiguous = False
            use_broadcast = True

        if b_sh == 1 and c_sh > 1:
            b._strides[i] = 0
            b._shape[i] = c_sh
            b._c_contiguous = b._f_contiguous = False
            use_broadcast = True

    if orig_a_ndim != 1:
        out_shape.append(a_part_outshape)
    if orig_b_ndim != 1:
        out_shape.append(b_part_outshape)

    # (A B)^T = B^T A^T
    a, b = b, a

    ka = a._shape[ndim - 2]
    lda = n = a._shape[ndim - 1]
    m = b._shape[ndim - 2]
    ldb = kb = b._shape[ndim - 1]

    if ka != kb:
        raise ValueError(
            'shapes ({}) and ({}) not aligned'.format(
                ','.join([str(_) for _ in orig_a.shape]),
                ','.join([str(_) for _ in orig_b.shape])))

    if out is not None and out.shape != tuple(out_shape):
        raise ValueError('Output array has an invalid size')

    if a.size == 0 or b.size == 0:
        if out is None:
            return cupy.zeros(out_shape, ret_dtype)
        else:
            out.fill(0)
            return out

    if (
        out is not None and out.dtype == dtype and out.flags.c_contiguous
        and not _memory_range.may_share_bounds(out, a)
        and not _memory_range.may_share_bounds(out, b)
    ):
        c = out
    else:
        c = core.ndarray(out_shape, dtype=dtype)
        if out is None:
            if dtype == ret_dtype:
                out = c
            else:
                out = core.ndarray(out_shape, dtype=ret_dtype)

    if orig_a_ndim == 1 or orig_b_ndim == 1:
        c_view = c.view()
        if orig_b_ndim == 1:
            c_view._shape.push_back(1)
            c_view._strides.push_back(0)
        if orig_a_ndim == 1:
            c_view._shape.insert(c_view._shape.end() - 1, 1)
            c_view._strides.insert(c_view._strides.end() - 1, 0)
        assert c_view._c_contiguous
        c_view._update_f_contiguity()
    else:
        c_view = c

    #cdef intptr_t handle = device.get_cublas_handle()
    #cdef int cuda_dtype = to_cuda_dtype(dtype)
    #cdef int algo = cublas.CUBLAS_GEMM_DEFAULT

    one = numpy.array(1, dtype=dtype)
    zero = numpy.array(0, dtype=dtype)
    if not use_broadcast:
        if dtype == numpy.float32:
            out = _ascend_matmul(a, b, out)
        else:
            raise TypeError(dtype, a.dtype, b.dtype)

    if out is not c:
        elementwise_copy(c, out)
    return out
