import numpy

import cupy
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import util


def batched_gesv(a, b):
    """Solves multiple linear matrix equations using cublas<t>getr[fs]Batched().

    Computes the solution to system of linear equation ``ax = b``.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(..., M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(..., M)`` or
            ``(..., M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(..., M)`` or ``(..., M, K)``.
    """
    util._assert_cupy_array(a, b)
    util._assert_nd_squareness(a)

    if not ((a.ndim == b.ndim or a.ndim == b.ndim + 1) and
            a.shape[:-1] == b.shape[:a.ndim - 1]):
        raise ValueError(
            'a must have (..., M, M) shape and b must have (..., M) '
            'or (..., M, K)')

    dtype = numpy.promote_types(a.dtype.char, 'f')
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise TypeError('invalid dtype')
    getrf = getattr(cublas, t + 'getrfBatched')
    getrs = getattr(cublas, t + 'getrsBatched')

    bs = numpy.prod(a.shape[:-2]) if a.ndim > 2 else 1
    n = a.shape[-1]
    nrhs = b.shape[-1] if a.ndim == b.ndim else 1
    b_shape = b.shape
    a_data_ptr = a.data.ptr
    b_data_ptr = b.data.ptr
    a = cupy.ascontiguousarray(a.reshape(bs, n, n).transpose(0, 2, 1),
                               dtype=dtype)
    b = cupy.ascontiguousarray(b.reshape(bs, n, nrhs).transpose(0, 2, 1),
                               dtype=dtype)
    if a.data.ptr == a_data_ptr:
        a = a.copy()
    if b.data.ptr == b_data_ptr:
        b = b.copy()

    handle = device.get_cublas_handle()
    lda = n
    a_step = lda * n * a.itemsize
    a_array = cupy.arange(a.data.ptr, a.data.ptr + a_step * bs, a_step,
                          dtype=cupy.uintp)
    ldb = n
    b_step = ldb * nrhs * b.itemsize
    b_array = cupy.arange(b.data.ptr, b.data.ptr + b_step * bs, b_step,
                          dtype=cupy.uintp)
    pivot = cupy.empty((bs, n), dtype=numpy.int32)
    info = cupy.empty((bs,), dtype=numpy.int32)
    # LU factorization (A = L * U)
    getrf(handle, n, a_array.data.ptr, lda, pivot.data.ptr, info.data.ptr, bs)
    util._check_cublas_info_array_if_synchronization_allowed(getrf, info)
    # Solves Ax = b
    getrs(handle, cublas.CUBLAS_OP_N, n, nrhs, a_array.data.ptr, lda,
          pivot.data.ptr, b_array.data.ptr, ldb, bs)
    return b.transpose(0, 2, 1).reshape(b_shape)
