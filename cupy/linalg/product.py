import numpy
import six

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy import elementwise


def dot(a, b, out=None, allocator=None):
    assert a.ndim > 0 and b.ndim > 0
    a_is_vec = a.ndim == 1
    b_is_vec = b.ndim == 1

    if a_is_vec:
        a = a.reshape(1, a.size)
    if b_is_vec:
        b = b.reshape(b.size, 1)
    c = tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)),
                  allocator=allocator, out=out)
    if out is None:
        if a_is_vec:
            if b_is_vec:
                c.shape = ()
            else:
                c.shape = c.shape[1:]
        elif b_is_vec:
            c.shape = c.shape[:-1]
    return c


def vdot(a, b):
    a = a.ravel()
    b = b.ravel()
    return a.dot(b)


def inner(a, b):
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    else:
        return tensordot(a, b, axes=(-1, -1))


def outer(a, b, out=None, allocator=None):
    a = a.reshape(a.size, 1)
    b = b.reshape(1, b.size)
    if out is None:
        return dot(a, b, allocator=allocator)
    elif out.flags.c_contiguous:
        return dot(a, b, out=out)
    else:
        out[:] = dot(a, b, allocator=allocator)
        return out


def tensordot(a, b, axes=2, allocator=None, out=None):
    if a.ndim == 0 or b.ndim == 0:
        if axes != 0:
            raise ValueError('An input is zero-dim while axes > 0')
        return cupy.multiply(a, b, out=out, allocator=allocator)

    ret_dtype = numpy.find_common_type([a.dtype, b.dtype], [])

    # Cast to float32 or float64
    dtype = numpy.find_common_type([a.dtype, b.dtype, 'f'], [])
    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    if a.dtype.type == numpy.float32:
        dot = cublas.sdot
        gemv = cublas.sgemv
        ger = cublas.sger
        gemm = cublas.sgemm
    elif a.dtype.type == numpy.float64:
        dot = cublas.ddot
        gemv = cublas.dgemv
        ger = cublas.dger
        gemm = cublas.dgemm

    if numpy.isscalar(axes):
        axes = [list(six.moves.range(a.ndim - axes, a.ndim)),
                list(six.moves.range(axes))]
    else:
        axes = list(axes)
    if numpy.isscalar(axes[0]):
        axes[0] = (axes[0],)
    if numpy.isscalar(axes[1]):
        axes[1] = (axes[1],)

    if len(axes) != 2:
        raise ValueError('Axes must consist of two arrays.')
    if len(axes[0]) != len(axes[1]):
        raise ValueError('Axes length mismatch')
    for a_axis, b_axis in zip(*axes):
        if not (-a.ndim <= a_axis < a.ndim and
                -b.ndim <= b_axis < b.ndim):
            raise IndexError('Axis overrun')
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('Axis dimension mismatch')

    # Make the axes non-negative
    axes = (tuple(axis % a.ndim for axis in axes[0]),
            tuple(axis % b.ndim for axis in axes[1]))

    sum_ndim = len(axes[0])
    a = _move_axes_to_head(a, axes[0])
    b = _move_axes_to_head(b, axes[1])

    m = numpy.prod(b.shape[sum_ndim:], dtype=numpy.int32)
    n = numpy.prod(a.shape[sum_ndim:], dtype=numpy.int32)
    ret_shape = a.shape[sum_ndim:] + b.shape[sum_ndim:]

    if out is not None:
        if out.size != numpy.prod(ret_shape, dtype=int):
            raise ValueError('Output array has an invalid size')
        if not out.flags.c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    if allocator is None:
        allocator = a.allocator
    if 0 in a.shape or 0 in b.shape:
        if 0 not in a.shape or 0 not in b.shape:
            raise ValueError('cannot dot zero-sized and non-zero-sized arrays')
        if out is None:
            return cupy.zeros(ret_shape, dtype=ret_dtype, allocator=allocator)
        else:
            out.fill(0)
            return out

    if out is None:
        out = cupy.empty(ret_shape, dtype=dtype, allocator=allocator)
        if dtype == ret_dtype:
            ret = out
        else:
            ret = cupy.empty(ret_shape, dtype=ret_dtype, allocator=allocator)
    else:
        ret = out
        if out.dtype != dtype:
            out = cupy.empty(ret_shape, dtype=dtype, allocator=allocator)

    k = a.size // n

    # It copies the operands if needed
    a = a.reshape(k, n)
    b = b.reshape(k, m).T

    a, transa = _mat_to_cublas_contiguous(a)
    b, transb = _mat_to_cublas_contiguous(b)

    lda = _get_leading_dim(a, transa)
    ldb = _get_leading_dim(b, transb)

    c = out.view()
    c.shape = (a.shape[1], b.shape[0])
    ldc = c.shape[1]

    handle = cuda.Device().cublas_handle
    if a.shape[1] == 1:
        lda = a.strides[0] // a.itemsize
        if b.shape[0] == 1:
            # inner product of vectors
            ldb = b.strides[1] // b.itemsize
            mode = cublas.getPointerMode(handle)
            cublas.setPointerMode(handle,
                                  cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dot(handle, k, a._fptr, lda, b._fptr, ldb, c._fptr)
            finally:
                cublas.setPointerMode(handle, mode)
        else:
            # B^T * A
            # sgemv requires (m, k) as the original matrix dimensions rather
            # than the transposed dimensions.
            if transb:
                m, k = k, m
            gemv(handle, transb, m, k, 1, b._fptr, ldb, a._fptr, lda,
                 0, c._fptr, 1)
    elif b.shape[0] == 1:
        # A * B
        ldb = b.strides[1] // b.itemsize
        # sgemv requires (n, k) as the original matrix dimensions rather
        # than the transposed dimensions.
        if not transa:
            n, k = k, n
        gemv(handle, 1 - transa, n, k, 1, a._fptr, lda, b._fptr, ldb,
             0, c._fptr, 1)
    elif a.shape[0] == 1:
        # B^T * A (outer product)
        assert b.shape[1] == 1
        c.fill(0)
        inca = a.strides[1] // a.itemsize
        incb = b.strides[0] // b.itemsize
        ger(handle, m, n, 1, b._fptr, incb, a._fptr, inca, c._fptr, ldc)
    else:
        # B^T * A^T
        gemm(handle, transb, transa, m, n, k, 1, b._fptr, ldb, a._fptr,
             lda, 0, c._fptr, ldc)

    if dtype != ret_dtype:
        elementwise.copy(out, ret)
    return ret


def einsum(subscripts, *args):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def matrix_power(M, n, allocator=cuda.alloc):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def kron(a, b, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def _move_axes_to_head(a, axes):
    # This function moves the axes of ``s`` to the head of the shape.
    axes = tuple(axes)
    right_axes = tuple(i for i in six.moves.range(a.ndim) if i not in axes)
    return a.transpose(*(axes + right_axes))


def _mat_to_cublas_contiguous(a):
    assert a.ndim == 2
    f = a.flags
    if f.c_contiguous:
        trans = cublas.CUBLAS_OP_T
    elif f.f_contiguous:
        trans = cublas.CUBLAS_OP_N
    else:
        a = a.copy()
        trans = cublas.CUBLAS_OP_T
    return a, trans


def _get_leading_dim(a, op):
    if op == cublas.CUBLAS_OP_N:
        return a.strides[1] // a.itemsize
    else:
        return a.strides[0] // a.itemsize
