import numpy as _numpy

import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device


def gesv(a, b):
    """Solve a linear matrix equation using cusolverDn<t>getr[fs]().

    Computes the solution to a system of linear equation ``ax = b``.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M)`` or ``(M, K)``.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim not in (1, 2):
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != a.shape[1]:
        raise ValueError('a must be a square matrix.')
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.
                         format(a.shape, b.shape))

    dtype = _numpy.promote_types(a.dtype.char, 'f')
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise ValueError('unsupported dtype (actual:{})'.format(a.dtype))
    helper = getattr(_cusolver, t + 'getrf_bufferSize')
    getrf = getattr(_cusolver, t + 'getrf')
    getrs = getattr(_cusolver, t + 'getrs')

    n = b.shape[0]
    nrhs = b.shape[1] if b.ndim == 2 else 1
    a_data_ptr = a.data.ptr
    b_data_ptr = b.data.ptr
    a = _cupy.asfortranarray(a, dtype=dtype)
    b = _cupy.asfortranarray(b, dtype=dtype)
    if a.data.ptr == a_data_ptr:
        a = a.copy()
    if b.data.ptr == b_data_ptr:
        b = b.copy()

    handle = _device.get_cusolver_handle()
    dipiv = _cupy.empty(n, dtype=_numpy.int32)
    dinfo = _cupy.empty(1, dtype=_numpy.int32)
    lwork = helper(handle, n, n, a.data.ptr, n)
    dwork = _cupy.empty(lwork, dtype=a.dtype)
    # LU factrization (A = L * U)
    getrf(handle, n, n, a.data.ptr, n, dwork.data.ptr, dipiv.data.ptr,
          dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        getrf, dinfo)
    # Solves Ax = b
    getrs(handle, _cublas.CUBLAS_OP_N, n, nrhs, a.data.ptr, n,
          dipiv.data.ptr, b.data.ptr, n, dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        getrs, dinfo)
    return b


def gels(a, b):
    """Solves over/well/under-determined linear systems.

    Computes least-square solution to equation ``ax = b` by QR factorization
    using cusolverDn<t>geqrf().

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, N)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(N)`` or ``(N, K)``.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim == 1:
        nrhs = 1
    elif b.ndim == 2:
        nrhs = b.shape[1]
    else:
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.
                         format(a.shape, b.shape))
    if a.dtype != b.dtype:
        raise ValueError('dtype mismatch (a: {}, b: {}).'.
                         format(a.dtype, b.dtype))

    dtype = a.dtype
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise ValueError('unsupported dtype (actual: {})'.format(dtype))

    geqrf_helper = getattr(_cusolver, t + 'geqrf_bufferSize')
    geqrf = getattr(_cusolver, t + 'geqrf')
    trsm = getattr(_cublas, t + 'trsm')
    if t in 'sd':
        ormqr_helper = getattr(_cusolver, t + 'ormqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'ormqr')
    else:
        ormqr_helper = getattr(_cusolver, t + 'unmqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'unmqr')

    no_trans = _cublas.CUBLAS_OP_N
    if dtype.char in 'fd':
        trans = _cublas.CUBLAS_OP_T
    else:
        trans = _cublas.CUBLAS_OP_C

    m, n = a.shape
    mn_min = min(m, n)
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    tau = _cupy.empty(mn_min, dtype=dtype)
    cusolver_handle = _device.get_cusolver_handle()
    cublas_handle = _device.get_cublas_handle()

    if m >= n:  # over/well-determined systems
        a = a.copy(order='F')
        b = b.copy(order='F')

        # geqrf (QR decomposition, A = Q * R)
        ws_size = geqrf_helper(cusolver_handle, m, n, a.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, m, n, a.data.ptr, m, tau.data.ptr,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            geqrf, dev_info)

        # ormqr (Computes Q^T * B)
        ws_size = ormqr_helper(
            cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs, mn_min,
            a.data.ptr, m, tau.data.ptr, b.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs,
              mn_min, a.data.ptr, m, tau.data.ptr, b.data.ptr, m,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            ormqr, dev_info)

        # trsm (Solves R * X = (Q^T * B))
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT,
             _cublas.CUBLAS_FILL_MODE_UPPER, no_trans,
             _cublas.CUBLAS_DIAG_NON_UNIT, mn_min, nrhs, 1, a.data.ptr, m,
             b.data.ptr, m)

        return b[:n]

    else:  # under-determined systems
        a = a.conj().T.copy(order='F')
        bb = b
        out_shape = (n,) if b.ndim == 1 else (n, nrhs)
        b = _cupy.zeros(out_shape, dtype=dtype, order='F')
        b[:m] = bb

        # geqrf (QR decomposition, A^T = Q * R)
        ws_size = geqrf_helper(cusolver_handle, n, m, a.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, n, m, a.data.ptr, n, tau.data.ptr,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            geqrf, dev_info)

        # trsm (Solves R^T * Z = B)
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT,
             _cublas.CUBLAS_FILL_MODE_UPPER, trans,
             _cublas.CUBLAS_DIAG_NON_UNIT, m, nrhs, 1, a.data.ptr, n,
             b.data.ptr, n)

        # ormqr (Computes Q * Z)
        ws_size = ormqr_helper(
            cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs,
            mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs,
              mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            ormqr, dev_info)

        return b
