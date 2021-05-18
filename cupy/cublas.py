import numpy
from numpy import linalg

import warnings

import cupy
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util

_batched_gesv_limit = 256


def get_batched_gesv_limit():
    global _batched_gesv_limit
    return _batched_gesv_limit


def set_batched_gesv_limit(limit):
    global _batched_gesv_limit
    _batched_gesv_limit = limit


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
    _util._assert_cupy_array(a, b)
    _util._assert_nd_squareness(a)

    if not ((a.ndim == b.ndim or a.ndim == b.ndim + 1) and
            a.shape[:-1] == b.shape[:a.ndim - 1]):
        raise ValueError(
            'a must have (..., M, M) shape and b must have (..., M) '
            'or (..., M, K)')

    dtype, out_dtype = _util.linalg_common_type(a, b)
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

    if n > get_batched_gesv_limit():
        warnings.warn('The matrix size ({}) exceeds the set limit ({})'.
                      format(n, get_batched_gesv_limit()))

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
    dinfo = cupy.empty((bs,), dtype=numpy.int32)
    info = numpy.empty((1,), dtype=numpy.int32)
    # LU factorization (A = L * U)
    getrf(handle, n, a_array.data.ptr, lda, pivot.data.ptr, dinfo.data.ptr, bs)
    _util._check_cublas_info_array_if_synchronization_allowed(getrf, dinfo)
    # Solves Ax = b
    getrs(handle, cublas.CUBLAS_OP_N, n, nrhs, a_array.data.ptr, lda,
          pivot.data.ptr, b_array.data.ptr, ldb, info.ctypes.data, bs)
    if info[0] != 0:
        msg = 'Error reported by {} in cuBLAS. '.format(getrs.__name__)
        if info[0] < 0:
            msg += 'The {}-th parameter had an illegal value.'.format(-info[0])
        raise linalg.LinAlgError(msg)

    return b.transpose(0, 2, 1).reshape(b_shape).astype(out_dtype, copy=False)


def iamax(x, out=None):
    """Finds the (smallest) index of the element with the maximum magnitude.

    Note: The result index is 1-based index (not 0-based index).
    """
    return _iamaxmin(x, out, 'amax')


def iamin(x, out=None):
    """Finds the (smallest) index of the element with the minimum magnitude.

    Note: The result index is 1-based index (not 0-based index).
    """
    return _iamaxmin(x, out, 'amin')


def _iamaxmin(x, out, name):
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))

    dtype = x.dtype.char
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
    func = getattr(cublas, 'i' + t + name)

    handle = device.get_cublas_handle()
    result_dtype = 'i'
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def asum(x, out=None):
    """Computes the sum of the absolute of x."""
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))

    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.sasum
    elif dtype == 'd':
        func = cublas.dasum
    elif dtype == 'F':
        func = cublas.scasum
    elif dtype == 'D':
        func = cublas.dzasum
    else:
        raise TypeError('invalid dtype')

    handle = device.get_cublas_handle()
    result_dtype = dtype.lower()
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def axpy(a, x, y):
    """Computes y += a * x.

    (*) y will be updated.
    """
    _check_two_vectors(x, y)

    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.saxpy
    elif dtype == 'd':
        func = cublas.daxpy
    elif dtype == 'F':
        func = cublas.caxpy
    elif dtype == 'D':
        func = cublas.zaxpy
    else:
        raise TypeError('invalid dtype')

    handle = device.get_cublas_handle()
    a, a_ptr, orig_mode = _setup_scalar_ptr(handle, a, dtype)
    try:
        func(handle, x.size, a_ptr, x.data.ptr, 1, y.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)


def dot(x, y, out=None):
    """Computes the dot product of x and y."""
    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.sdot
    elif dtype == 'd':
        func = cublas.ddot
    elif dtype in 'FD':
        raise TypeError('Use dotu() or dotc() for complex dtype')
    else:
        raise TypeError('invalid dtype')
    _check_two_vectors(x, y)

    handle = device.get_cublas_handle()
    result_dtype = dtype
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, y.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def dotu(x, y, out=None):
    """Computes the dot product of x and y."""
    dtype = x.dtype.char
    if dtype in 'fd':
        return dot(x, y, out=out)
    elif dtype == 'F':
        func = cublas.cdotu
    elif dtype == 'D':
        func = cublas.zdotu
    else:
        raise TypeError('invalid dtype')
    _check_two_vectors(x, y)

    handle = device.get_cublas_handle()
    result_dtype = dtype
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, y.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def dotc(x, y, out=None):
    """Computes the dot product of x.conj() and y."""
    dtype = x.dtype.char
    if dtype in 'fd':
        return dot(x, y, out=out)
    elif dtype == 'F':
        func = cublas.cdotc
    elif dtype == 'D':
        func = cublas.zdotc
    else:
        raise TypeError('invalid dtype')
    _check_two_vectors(x, y)

    handle = device.get_cublas_handle()
    result_dtype = dtype
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, y.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def nrm2(x, out=None):
    """Computes the Euclidean norm of vector x."""
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))

    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.snrm2
    elif dtype == 'd':
        func = cublas.dnrm2
    elif dtype == 'F':
        func = cublas.scnrm2
    elif dtype == 'D':
        func = cublas.dznrm2
    else:
        raise TypeError('invalid dtype')

    handle = device.get_cublas_handle()
    result_dtype = dtype.lower()
    result_ptr, result, orig_mode = _setup_result_ptr(
        handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)

    if out is None:
        out = result
    elif out.dtype != result_dtype:
        out[...] = result
    return out


def scal(a, x):
    """Computes x *= a.

    (*) x will be updated.
    """
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))

    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.sscal
    elif dtype == 'd':
        func = cublas.dscal
    elif dtype == 'F':
        func = cublas.cscal
    elif dtype == 'D':
        func = cublas.zscal
    else:
        raise TypeError('invalid dtype')

    handle = device.get_cublas_handle()
    a, a_ptr, orig_mode = _setup_scalar_ptr(handle, a, dtype)
    try:
        func(handle, x.size, a_ptr, x.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)


def _check_two_vectors(x, y):
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))
    if y.ndim != 1:
        raise ValueError('y must be a 1D array (actual: {})'.format(y.ndim))
    if x.size != y.size:
        raise ValueError('x and y must be the same size (actual: {} and {})'
                         ''.format(x.size, y.size))
    if x.dtype != y.dtype:
        raise TypeError('x and y must be the same dtype (actual: {} and {})'
                        ''.format(x.dtype, y.dtype))


def _setup_result_ptr(handle, out, dtype):
    mode = cublas.getPointerMode(handle)
    if out is None or isinstance(out, cupy.ndarray):
        if out is None or out.dtype != dtype:
            result = cupy.empty([], dtype=dtype)
        else:
            result = out
        result_ptr = result.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    elif isinstance(out, numpy.ndarray):
        if out.dtype != dtype:
            result = numpy.empty([], dtype=dtype)
        else:
            result = out
        result_ptr = result.ctypes.data
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    else:
        raise TypeError('out must be either cupy or numpy ndarray')
    return result_ptr, result, mode


def _setup_scalar_ptr(handle, a, dtype):
    a, a_ptr = _get_scalar_ptr(a, dtype)
    mode = cublas.getPointerMode(handle)
    if isinstance(a, cupy.ndarray):
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    return a, a_ptr, mode


def _get_scalar_ptr(a, dtype):
    if isinstance(a, cupy.ndarray):
        if a.dtype != dtype:
            a = cupy.array(a, dtype=dtype)
        a_ptr = a.data.ptr
    else:
        if not (isinstance(a, numpy.ndarray) and a.dtype == dtype):
            a = numpy.array(a, dtype=dtype)
        a_ptr = a.ctypes.data
    return a, a_ptr


def gemv(transa, alpha, a, x, beta, y):
    """Computes y = alpha * op(a) @ x + beta * y

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.

    Note: ''y'' will be updated.
    """
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemv
    elif dtype == 'd':
        func = cublas.dgemv
    elif dtype == 'F':
        func = cublas.cgemv
    elif dtype == 'D':
        func = cublas.zgemv
    else:
        raise TypeError('invalid dtype')
    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    transa = _trans_to_cublas_op(transa)
    if transa == cublas.CUBLAS_OP_N:
        xlen, ylen = n, m
    else:
        xlen, ylen = m, n
    assert x.shape[0] == xlen
    assert y.shape[0] == ylen

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    try:
        if a._f_contiguous:
            func(handle, transa, m, n, alpha_ptr, a.data.ptr, m, x.data.ptr, 1,
                 beta_ptr, y.data.ptr, 1)
        elif a._c_contiguous and transa != cublas.CUBLAS_OP_C:
            if transa == cublas.CUBLAS_OP_N:
                transa = cublas.CUBLAS_OP_T
            else:
                transa = cublas.CUBLAS_OP_N
            func(handle, transa, n, m, alpha_ptr, a.data.ptr, n, x.data.ptr, 1,
                 beta_ptr, y.data.ptr, 1)
        else:
            a = a.copy(order='F')
            func(handle, transa, m, n, alpha_ptr, a.data.ptr, m, x.data.ptr, 1,
                 beta_ptr, y.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)


def ger(alpha, x, y, a):
    """Computes a += alpha * x @ y.T

    Note: ''a'' will be updated.
    """
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sger
    elif dtype == 'd':
        func = cublas.dger
    elif dtype in 'FD':
        raise TypeError('Use geru or gerc for complex dtypes')
    else:
        raise TypeError('invalid dtype')

    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    assert x.shape[0] == m
    assert y.shape[0] == n

    handle = device.get_cublas_handle()
    alpha, alpha_ptr, orig_mode = _setup_scalar_ptr(handle, alpha, dtype)
    x_ptr, y_ptr = x.data.ptr, y.data.ptr
    try:
        if a._f_contiguous:
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, a.data.ptr, m)
        elif a._c_contiguous:
            func(handle, n, m, alpha_ptr, y_ptr, 1, x_ptr, 1, a.data.ptr, n)
        else:
            aa = a.copy(order='F')
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, aa.data.ptr, m)
            a[...] = aa
    finally:
        cublas.setPointerMode(handle, orig_mode)


def geru(alpha, x, y, a):
    """Computes a += alpha * x @ y.T

    Note: ''a'' will be updated.
    """
    dtype = a.dtype.char
    if dtype in 'fd':
        return ger(alpha, x, y, a)
    elif dtype == 'F':
        func = cublas.cgeru
    elif dtype == 'D':
        func = cublas.zgeru
    else:
        raise TypeError('invalid dtype')
    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    assert x.shape[0] == m
    assert y.shape[0] == n

    handle = device.get_cublas_handle()
    alpha, alpha_ptr, orig_mode = _setup_scalar_ptr(handle, alpha, dtype)
    x_ptr, y_ptr = x.data.ptr, y.data.ptr
    try:
        if a._f_contiguous:
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, a.data.ptr, m)
        elif a._c_contiguous:
            func(handle, n, m, alpha_ptr, y_ptr, 1, x_ptr, 1, a.data.ptr, n)
        else:
            aa = a.copy(order='F')
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, aa.data.ptr, m)
            a[...] = aa
    finally:
        cublas.setPointerMode(handle, orig_mode)


def gerc(alpha, x, y, a):
    """Computes a += alpha * x @ y.T.conj()

    Note: ''a'' will be updated.
    """
    dtype = a.dtype.char
    if dtype in 'fd':
        return ger(alpha, x, y, a)
    elif dtype == 'F':
        func = cublas.cgerc
    elif dtype == 'D':
        func = cublas.zgerc
    else:
        raise TypeError('invalid dtype')
    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    assert x.shape[0] == m
    assert y.shape[0] == n

    handle = device.get_cublas_handle()
    alpha, alpha_ptr, orig_mode = _setup_scalar_ptr(handle, alpha, dtype)
    x_ptr, y_ptr = x.data.ptr, y.data.ptr
    try:
        if a._f_contiguous:
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, a.data.ptr, m)
        else:
            aa = a.copy(order='F')
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, aa.data.ptr, m)
            a[...] = aa
    finally:
        cublas.setPointerMode(handle, orig_mode)


def _trans_to_cublas_op(trans):
    if trans == 'N' or trans == cublas.CUBLAS_OP_N:
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T' or trans == cublas.CUBLAS_OP_T:
        trans = cublas.CUBLAS_OP_T
    elif trans == 'H' or trans == cublas.CUBLAS_OP_C:
        trans = cublas.CUBLAS_OP_C
    else:
        raise TypeError('invalid trans (actual: {})'.fromat(trans))
    return trans


def _decide_ld_and_trans(a, trans):
    ld = None
    if trans in (cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T):
        if a._f_contiguous:
            ld = a.shape[0]
        elif a._c_contiguous:
            ld = a.shape[1]
            trans = 1 - trans
    return ld, trans


def _change_order_if_necessary(a, lda):
    if lda is None:
        lda = a.shape[0]
        if not a._f_contiguous:
            a = a.copy(order='F')
    return a, lda


def gemm(transa, transb, a, b, out=None, alpha=1.0, beta=0.0):
    """Computes out = alpha * op(a) @ op(b) + beta * out

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'H'.
    """
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemm
    elif dtype == 'd':
        func = cublas.dgemm
    elif dtype == 'F':
        func = cublas.cgemm
    elif dtype == 'D':
        func = cublas.zgemm
    else:
        raise TypeError('invalid dtype')

    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, k = a.shape
    else:
        k, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        n = b.shape[1]
        assert b.shape[0] == k
    else:
        n = b.shape[0]
        assert b.shape[1] == k
    if out is None:
        out = cupy.empty((m, n), dtype=dtype, order='F')
        beta = 0.0
    else:
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, k, alpha_ptr,
                     a.data.ptr, lda, b.data.ptr, ldb, beta_ptr, out.data.ptr,
                     m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            # Computes out.T = alpha * b.T @ a.T + beta * out.T
            try:
                func(handle, 1 - transb, 1 - transa, n, m, k, alpha_ptr,
                     b.data.ptr, ldb, a.data.ptr, lda, beta_ptr, out.data.ptr,
                     n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out

    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr, lda,
             b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        out[...] = c
    return out


def geam(transa, transb, alpha, a, beta, b, out=None):
    """Computes alpha * op(a) + beta * op(b)

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'H'.
    """
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgeam
    elif dtype == 'd':
        func = cublas.dgeam
    elif dtype == 'F':
        func = cublas.cgeam
    elif dtype == 'D':
        func = cublas.zgeam
    else:
        raise TypeError('invalid dtype')

    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, n = a.shape
    else:
        n, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        assert b.shape == (m, n)
    else:
        assert b.shape == (n, m)
    if out is None:
        out = cupy.empty((m, n), dtype=dtype, order='F')
    else:
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, alpha_ptr, a.data.ptr,
                     lda, beta_ptr, b.data.ptr, ldb, out.data.ptr, m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            # Computes alpha * a.T + beta * b.T
            try:
                func(handle, 1-transa, 1-transb, n, m, alpha_ptr, a.data.ptr,
                     lda, beta_ptr, b.data.ptr, ldb, out.data.ptr, n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out

    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, alpha_ptr, a.data.ptr, lda,
             beta_ptr, b.data.ptr, ldb, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        out[...] = c
    return out


def dgmm(side, a, x, out=None, incx=1):
    """Computes diag(x) @ a or a @ diag(x)

    Computes diag(x) @ a if side is 'L', a @ diag(x) if side is 'R'.
    """
    assert a.ndim == 2
    assert 0 <= x.ndim <= 2
    assert a.dtype == x.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sdgmm
    elif dtype == 'd':
        func = cublas.ddgmm
    elif dtype == 'F':
        func = cublas.cdgmm
    elif dtype == 'D':
        func = cublas.zdgmm
    else:
        raise TypeError('invalid dtype')
    if side == 'L' or side == cublas.CUBLAS_SIDE_LEFT:
        side = cublas.CUBLAS_SIDE_LEFT
    elif side == 'R' or side == cublas.CUBLAS_SIDE_RIGHT:
        side = cublas.CUBLAS_SIDE_RIGHT
    else:
        raise ValueError('invalid side (actual: {})'.format(side))
    m, n = a.shape
    if side == cublas.CUBLAS_SIDE_LEFT:
        assert x.size >= (m - 1) * abs(incx) + 1
    else:
        assert x.size >= (n - 1) * abs(incx) + 1
    if out is None:
        if a._c_contiguous:
            order = 'C'
        else:
            order = 'F'
        out = cupy.empty((m, n), dtype=dtype, order=order)
    else:
        assert out.ndim == 2
        assert out.shape == a.shape
        assert out.dtype == a.dtype

    handle = device.get_cublas_handle()
    if out._c_contiguous:
        if not a._c_contiguous:
            a = a.copy(order='C')
        func(handle, 1 - side, n, m, a.data.ptr, n, x.data.ptr, incx,
             out.data.ptr, n)
    else:
        if not a._f_contiguous:
            a = a.copy(order='F')
        c = out
        if not out._f_contiguous:
            c = out.copy(order='F')
        func(handle, side, m, n, a.data.ptr, m, x.data.ptr, incx,
             c.data.ptr, m)
        if not out._f_contiguous:
            out[...] = c
    return out
