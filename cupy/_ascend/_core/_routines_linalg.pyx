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
from cupy.backends.backend.api cimport runtime
from cupy.backends.ascend.api.acl_utils cimport launch_general_func

cdef extern from '../../../cupy/backends/cupy_complex.h':
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

# TODO(ASCEND) ComputeType maybe diff for diff xpu backend
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

cdef _ndarray_base _ascend_dot(_ndarray_base a, _ndarray_base b, _ndarray_base out):
    """Dispatch `ascend_dot` (aclnnDot).

    aclnnDot requires two 1-D tensors of equal shape and a 0-D output.
    Callers are responsible for reshaping to that contract.
    """
    launch_general_func("ascend_dot", [a, b], [out], [], {}, 0)
    return out

cpdef _ndarray_base dot(_ndarray_base a, _ndarray_base b, _ndarray_base out=None):
    # share all the code with cupy' orignal version
    cdef Py_ssize_t a_ndim, b_ndim, a_axis, b_axis, n, m, k
    cdef bint input_a_is_vec, input_b_is_vec
    cdef shape_t ret_shape, shape

    a_ndim = a._shape.size()
    b_ndim = b._shape.size()

    if out is not None:
        if numpy.result_type(a.dtype, b.dtype) != out.dtype:
            raise ValueError('Not supported dtype combination.')
        if not out._c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    if a_ndim == 0 or b_ndim == 0:
        return _math._multiply(a, b, out=out)

    input_a_is_vec = a_ndim == 1
    input_b_is_vec = b_ndim == 1
    IF CUPY_CANN_VERSION > 0:
        # ------------------------------------------------------------------
        # ASCEND: dispatch numpy.dot semantics onto the available aclnn ops.
        #
        #   aclnnDot    : 1-D . 1-D  -> 0-D scalar
        #   aclnnMatmul : 2-D @ 2-D  -> 2-D matrix
        #
        # Other combinations (mixed vector/matrix, ndim > 2) are not directly
        # supported by these two aclnn ops; route them through the generic
        # `tensordot_core` helper below which handles the general case.
        # ------------------------------------------------------------------
        if a_ndim == 1 and b_ndim == 1:
            # dot product -> scalar; tensordot_core handles out allocation
            if a.size != b.size:
                raise ValueError(
                    'shapes ({},) and ({},) not aligned'.format(
                        a.size, b.size))
            ret_shape.clear()
            ret_shape.push_back(1)
            return tensordot_core(a, b, out, 1, 1, a.size, ret_shape)

        if a_ndim == 2 and b_ndim == 2:
            return _ascend_matmul(a, b, out)

        # (n,) . (n, ...)  -> (...):  reshape to (1, n) @ (n, tail)
        if a_ndim == 1:
            if a.size != b._shape[b_ndim - 2]:
                raise ValueError('Axis dimension mismatch')
            tail = 1
            for i in range(b_ndim):
                if i != b_ndim - 2:
                    tail *= b._shape[i]
            prod = _ascend_matmul(
                _manipulation._reshape(a, [1, a.size]),
                _manipulation._reshape(b, [b._shape[b_ndim - 2], tail]),
                None)
            result = _manipulation._reshape(prod, b._shape[1:])
            if out is not None:
                elementwise_copy(result, out)
                return out
            return result

        # (..., n) . (n,)  -> (...):  reshape to (m, n) @ (n, 1)
        if b_ndim == 1:
            if a._shape[a_ndim - 1] != b.size:
                raise ValueError('Axis dimension mismatch')
            m = a.size // a._shape[a_ndim - 1]
            prod = _ascend_matmul(
                _manipulation._reshape(a, [m, a._shape[a_ndim - 1]]),
                _manipulation._reshape(b, [b.size, 1]),
                None)
            result = _manipulation._reshape(prod, a._shape[:a_ndim - 1])
            if out is not None:
                elementwise_copy(result, out)
                return out
            return result

        raise NotImplementedError(
            "ASCEND: dot() supports 1-D/2-D operands; got ndim {} and {}"
            .format(a_ndim, b_ndim))
        # unreachable generic path below is CUDA-only
    ELSE:
        if input_a_is_vec:
            shape.clear()
            shape.push_back(1)
            shape.push_back(a.size)
            a = _manipulation._reshape(a, shape)
            a_ndim = 2
        if input_b_is_vec:
            shape.clear()
            shape.push_back(b.size)
            shape.push_back(1)
            b = _manipulation._reshape(b, shape)
            b_ndim = 2

        a_axis = a_ndim - 1
        b_axis = b_ndim - 2

        if a._shape[a_axis] != b._shape[b_axis]:
            raise ValueError('Axis dimension mismatch')

        if a_axis:
            a = _manipulation.rollaxis(a, a_axis, 0)
        if b_axis:
            b = _manipulation.rollaxis(b, b_axis, 0)

        k = a._shape[0]
        if k != 0:
            m = b.size // k
            n = a.size // k
        else:
            # When k==0, the function must return a matrix filled with zero
            # like NumPy.
            m = 0
            n = 0

        if not input_a_is_vec:
            ret_shape.insert(
                ret_shape.end(), a._shape.begin() + 1, a._shape.end())
        if not input_b_is_vec:
            ret_shape.insert(
                ret_shape.end(), b._shape.begin() + 1, b._shape.end())
        if out is not None:
            # TODO(kataoka): Make the condition strict
            if k != 0 and out.size != n * m:
                raise ValueError('Output array has an invalid size')

        return tensordot_core(a, b, out, n, m, k, ret_shape)

    # The ASCEND branch above always returns or raises; the fall-through
    # below is unreachable for ascend.
    return None

cpdef _ndarray_base tensordot_core(
        _ndarray_base a, _ndarray_base b, _ndarray_base out, Py_ssize_t n,
        Py_ssize_t m, Py_ssize_t k, const shape_t& ret_shape):
    # out, if specified, must be C-contiguous and have correct shape.
    cdef shape_t shape
    #cdef Py_ssize_t transa, transb, lda, ldb
    #cdef intptr_t handle
    cdef _ndarray_base copy_to_out = None
    cdef _ndarray_base a_vec, b_vec, scalar_out
    cdef str dtype = a.dtype.char
    #cdef int compute_capability = int(device.get_compute_capability())
    if dtype != b.dtype.char:
        dtype = numpy.promote_types(dtype, b.dtype).char
    if not a.size or not b.size:
        if out is None:
            out = _ndarray_init(cupy.ndarray, ret_shape, dtype, None)
        out.fill(0)
        return out

    if out is not None:
        assert out.flags.c_contiguous and out.dtype == dtype
    else:
        out = _ndarray_init(cupy.ndarray, ret_shape, dtype, None)

    IF CUPY_CANN_VERSION > 0:
        # ASCEND: aclnnDot only accepts two 1-D tensors and writes a 0-D
        # output. Flatten both operands to 1-D and use a temporary 0-D array,
        # then copy back into `out`.
        if (a._shape.size() == 1 and b._shape.size() == 1
                and a.size == b.size):
            a_vec = a
            b_vec = b
            scalar_out = _ndarray_init(
                cupy.ndarray, shape, dtype, None)  # shape == () -> 0-D
            launch_general_func(
                "ascend_dot", [a_vec, b_vec], [scalar_out], [], {}, 0)
            elementwise_copy(scalar_out, out)
            return out
        raise NotImplementedError(
            "ASCEND: tensordot/dot for ndim>1 non-matrix case is not "
            "supported, use matmul() instead")

    _ascend_dot(a, b, out)
    return out

cdef _ndarray_base _ascend_batched_matmul(
        _ndarray_base a, _ndarray_base b, _ndarray_base out,
        int a_ndim, int b_ndim):
    """numpy.matmul semantics for ndim > 2 (or with 1-D operands).

    aclnnMatmul only handles 2-D operands, so the batch dimensions are folded
    into the leading dimension and the batches are broadcast explicitly.
    """
    cdef int ndim = max(a_ndim, b_ndim)
    cdef _ndarray_base a2, b2, prod, result
    cdef list batch_shape = []
    cdef list a_shape, b_shape
    cdef Py_ssize_t i, a_batch, b_batch, batch, n, k, m
    cdef bint a_is_vec, b_is_vec

    a_is_vec = a_ndim == 1
    b_is_vec = b_ndim == 1

    if a_is_vec:
        raise NotImplementedError(
            'ASCEND: matmul with a 1-D operand is not supported')
    if b_ndim < 2:
        raise NotImplementedError(
            'ASCEND: matmul with a 1-D operand is not supported')

    # broadcast the batch shapes
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    for i in range(ndim - 2):
        da = a_shape[i] if i < a_ndim - 2 else 1
        db = b_shape[i] if i < b_ndim - 2 else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(
                'operands could not be broadcast together with remapped '
                'shapes {} and {}'.format(tuple(a_shape), tuple(b_shape)))
        batch_shape.append(max(da, db))

    n = a_shape[a_ndim - 2]
    k = a_shape[a_ndim - 1]
    if k != b_shape[b_ndim - 2]:
        raise ValueError(
            'matmul: shape mismatch, {} != {}'
            .format(k, b_shape[b_ndim - 2]))
    m = b_shape[b_ndim - 1]

    batch = 1
    for i in range(len(batch_shape)):
        batch *= batch_shape[i]

    # normalise to (batch, n, k) and (batch, k, m) without copying when the
    # leading batch dimensions are already explicit.
    a_batch = 1
    for i in range(a_ndim - 2):
        a_batch *= a_shape[i]
    b_batch = 1
    for i in range(b_ndim - 2):
        b_batch *= b_shape[i]

    a2 = _manipulation._reshape(a, [a_batch, n, k])
    b2 = _manipulation._reshape(b, [b_batch, k, m])

    if a_batch != batch:
        a2 = _manipulation.broadcast_to(
            a2, [batch, n, k]).copy()
    if b_batch != batch:
        b2 = _manipulation.broadcast_to(
            b2, [batch, k, m]).copy()

    # fold the batch dimension into the row dimension: (batch*n, k) @ (k, m)
    # is not equivalent for each batch, so loop over the batch instead.
    result = _ndarray_init(
        cupy.ndarray, batch_shape + [n, m],
        numpy.promote_types(a.dtype, b.dtype), None)
    result_flat = _manipulation._reshape(result, [batch, n, m])
    a_flat = _manipulation._reshape(a2, [batch, n, k])
    b_flat = _manipulation._reshape(b2, [batch, k, m])
    for i in range(batch):
        _ascend_matmul(
            _manipulation._reshape(a_flat[i], [n, k]),
            _manipulation._reshape(b_flat[i], [k, m]),
            _manipulation._reshape(result_flat[i], [n, m]))

    if out is not None:
        elementwise_copy(result, out)
        return out
    return result


cdef _ndarray_base _ascend_matmul(_ndarray_base a, _ndarray_base b, _ndarray_base out):
    """2-D matrix multiply via aclnnMatmul: out = a @ b.

    aclnnMatmul is a plain row-major A @ B (no cuBLAS column-major
    transpose trick), so operands must NOT be swapped here.
    """
    if out is None:
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                'shapes ({}) and ({}) not aligned'.format(a.shape, b.shape))
        ret_shape = [a.shape[0], b.shape[1]]
        ret_dtype = numpy.promote_types(a.dtype, b.dtype)
        out = _ndarray_init(cupy.ndarray, ret_shape, ret_dtype, None)
    launch_general_func("ascend_matmul", [a, b], [out], [], {}, 0)
    return out

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
    #from cupy.backends.backend.libs import cublas

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

    IF CUPY_CANN_VERSION <= 0:
        if ndim <= 2:
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
        # ASCEND: aclnnMatmul already computes A @ B directly, no cuBLAS
        # "transpose trick" is needed. Return immediately, otherwise the code
        # below (which swaps a/b for the cuBLAS column-major convention) would
        # recompute B @ A and silently produce a transposed result.
        if ndim == 2 and orig_a_ndim == 2 and orig_b_ndim == 2:
            # plain 2-D @ 2-D
            return _ascend_matmul(a, b, out)

        return _ascend_batched_matmul(a, b, out, orig_a_ndim, orig_b_ndim)

    # ===================================================================
    # The block below is the CUDA/cuBLAS path. ASCEND never reaches here
    # because the `ELSE` branch above always returns (ndim == 2) or raises
    # (ndim != 2). It is kept (guarded) for the CUDA backend only.
    # ===================================================================
    IF CUPY_CANN_VERSION <= 0:
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

        if out is not c:
            elementwise_copy(c, out)
        return out
