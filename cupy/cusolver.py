import numpy as _numpy

import cupy as _cupy
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device
from cupy import _util

_available_cuda_version = {
    'gesvdj': (9000, None),
    'gesvda': (10010, None),
    'potrfBatched': (9010, None),
    'potrsBatched': (9010, None),
    'syevj': (9000, None),
}


@_util.memoize()
def check_availability(name):
    if name not in _available_cuda_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = _available_cuda_version[name]
    cuda_version = _runtime.runtimeGetVersion()
    if version_added is not None and cuda_version < version_added:
        return False
    if version_removed is not None and cuda_version >= version_removed:
        return False
    return True


def gesvdj(a, full_matrices=True, compute_uv=True, overwrite_a=False):
    """Singular value decomposition using cusolverDn<t>gesvdj().

    Factorizes the matrix ``a`` into two unitary matrices ``u`` and ``v`` and
    a singular values vector ``s`` such that ``a == u @ diag(s) @ v*``.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(M, N)``.
        full_matrices (bool): If True, it returns u and v with dimensions
            ``(M, M)`` and ``(N, N)``. Otherwise, the dimensions of u and v
            are respectively ``(M, K)`` and ``(K, N)``, where
            ``K = min(M, N)``.
        compute_uv (bool): If ``False``, it only returns singular values.
        overwrite_a (bool): If ``True``, matrix ``a`` might be overwritten.

    Returns:
        tuple of :class:`cupy.ndarray`:
            A tuple of ``(u, s, v)``.
    """
    if not check_availability('gesvdj'):
        raise RuntimeError('gesvdj is not available.')

    if a.ndim == 3:
        return _gesvdj_batched(a, full_matrices, compute_uv, overwrite_a)

    assert a.ndim == 2

    if a.dtype == 'f':
        helper = _cusolver.sgesvdj_bufferSize
        solver = _cusolver.sgesvdj
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = _cusolver.dgesvdj_bufferSize
        solver = _cusolver.dgesvdj
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = _cusolver.cgesvdj_bufferSize
        solver = _cusolver.cgesvdj
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = _cusolver.zgesvdj_bufferSize
        solver = _cusolver.zgesvdj
        s_dtype = 'd'
    else:
        raise TypeError

    handle = _device.get_cusolver_handle()
    m, n = a.shape
    a = _cupy.array(a, order='F', copy=not overwrite_a)
    lda = m
    mn = min(m, n)
    s = _cupy.empty(mn, dtype=s_dtype)
    ldu = m
    ldv = n
    if compute_uv:
        jobz = _cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = _cusolver.CUSOLVER_EIG_MODE_NOVECTOR
        full_matrices = False
    if full_matrices:
        econ = 0
        u = _cupy.empty((ldu, m), dtype=a.dtype, order='F')
        v = _cupy.empty((ldv, n), dtype=a.dtype, order='F')
    else:
        econ = 1
        u = _cupy.empty((ldu, mn), dtype=a.dtype, order='F')
        v = _cupy.empty((ldv, mn), dtype=a.dtype, order='F')
    params = _cusolver.createGesvdjInfo()
    lwork = helper(handle, jobz, econ, m, n, a.data.ptr, lda, s.data.ptr,
                   u.data.ptr, ldu, v.data.ptr, ldv, params)
    work = _cupy.empty(lwork, dtype=a.dtype)
    info = _cupy.empty(1, dtype=_numpy.int32)
    solver(handle, jobz, econ, m, n, a.data.ptr, lda, s.data.ptr,
           u.data.ptr, ldu, v.data.ptr, ldv, work.data.ptr, lwork,
           info.data.ptr, params)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvdj, info)

    _cusolver.destroyGesvdjInfo(params)
    if compute_uv:
        return u, s, v
    else:
        return s


def _gesvdj_batched(a, full_matrices, compute_uv, overwrite_a):
    if a.dtype == 'f':
        helper = _cusolver.sgesvdjBatched_bufferSize
        solver = _cusolver.sgesvdjBatched
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = _cusolver.dgesvdjBatched_bufferSize
        solver = _cusolver.dgesvdjBatched
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = _cusolver.cgesvdjBatched_bufferSize
        solver = _cusolver.cgesvdjBatched
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = _cusolver.zgesvdjBatched_bufferSize
        solver = _cusolver.zgesvdjBatched
        s_dtype = 'd'
    else:
        raise TypeError

    handle = _device.get_cusolver_handle()
    batch_size, m, n = a.shape
    a = _cupy.array(a.swapaxes(-2, -1), order='C', copy=not overwrite_a)
    lda = m
    mn = min(m, n)
    s = _cupy.empty((batch_size, mn), dtype=s_dtype)
    ldu = m
    ldv = n
    if compute_uv:
        jobz = _cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = _cusolver.CUSOLVER_EIG_MODE_NOVECTOR
        # if not batched, `full_matrices = False` could speedup.

    u = _cupy.empty((batch_size, m, ldu), dtype=a.dtype).swapaxes(-2, -1)
    v = _cupy.empty((batch_size, n, ldv), dtype=a.dtype).swapaxes(-2, -1)
    params = _cusolver.createGesvdjInfo()
    lwork = helper(handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
                   u.data.ptr, ldu, v.data.ptr, ldv, params, batch_size)
    work = _cupy.empty(lwork, dtype=a.dtype)
    info = _cupy.empty(1, dtype=_numpy.int32)
    solver(handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
           u.data.ptr, ldu, v.data.ptr, ldv, work.data.ptr, lwork,
           info.data.ptr, params, batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvdj, info)

    _cusolver.destroyGesvdjInfo(params)
    if not full_matrices:
        u = u[..., :mn]
        v = v[..., :mn]
    if compute_uv:
        return u, s, v
    else:
        return s


def gesvda(a, compute_uv=True):
    """Singular value decomposition using cusolverDn<t>gesvdaStridedBatched().

    Factorizes the matrix ``a`` into two unitary matrices ``u`` and ``v`` and
    a singular values vector ``s`` such that ``a == u @ diag(s) @ v*``.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(.., M, N)``.
        compute_uv (bool): If ``False``, it only returns singular values.

    Returns:
        tuple of :class:`cupy.ndarray`:
            A tuple of ``(u, s, v)``.
    """
    if not check_availability('gesvda'):
        raise RuntimeError('gesvda is not available.')

    assert a.ndim >= 2
    a_ndim = a.ndim
    a_shape = a.shape
    m, n = a_shape[-2:]
    assert m >= n

    if a.dtype == 'f':
        helper = _cusolver.sgesvdaStridedBatched_bufferSize
        solver = _cusolver.sgesvdaStridedBatched
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = _cusolver.dgesvdaStridedBatched_bufferSize
        solver = _cusolver.dgesvdaStridedBatched
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = _cusolver.cgesvdaStridedBatched_bufferSize
        solver = _cusolver.cgesvdaStridedBatched
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = _cusolver.zgesvdaStridedBatched_bufferSize
        solver = _cusolver.zgesvdaStridedBatched
        s_dtype = 'd'
    else:
        raise TypeError

    handle = _device.get_cusolver_handle()
    if compute_uv:
        jobz = _cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = _cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    rank = min(m, n)
    if a_ndim == 2:
        batch_size = 1
    else:
        batch_size = _numpy.array(a_shape[:-2]).prod().item()
    a = a.reshape((batch_size, m, n))
    a = _cupy.ascontiguousarray(a.transpose(0, 2, 1))
    lda = m
    stride_a = lda * n
    s = _cupy.empty((batch_size, rank), dtype=s_dtype)
    stride_s = rank
    ldu = m
    ldv = n
    u = _cupy.empty((batch_size, rank, ldu), dtype=a.dtype, order='C')
    v = _cupy.empty((batch_size, rank, ldv), dtype=a.dtype, order='C')
    stride_u = rank * ldu
    stride_v = rank * ldv
    lwork = helper(handle, jobz, rank, m, n, a.data.ptr, lda, stride_a,
                   s.data.ptr, stride_s, u.data.ptr, ldu, stride_u,
                   v.data.ptr, ldv, stride_v, batch_size)
    work = _cupy.empty((lwork,), dtype=a.dtype)
    info = _cupy.empty((batch_size,), dtype=_numpy.int32)
    r_norm = _numpy.empty((batch_size,), dtype=_numpy.float64)
    solver(handle, jobz, rank, m, n, a.data.ptr, lda, stride_a, s.data.ptr,
           stride_s, u.data.ptr, ldu, stride_u, v.data.ptr, ldv, stride_v,
           work.data.ptr, lwork, info.data.ptr, r_norm.ctypes.data, batch_size)

    s = s.reshape(a_shape[:-2] + (s.shape[-1],))
    if not compute_uv:
        return s

    u = u.transpose(0, 2, 1)
    v = v.transpose(0, 2, 1)
    u = u.reshape(a_shape[:-2] + (u.shape[-2:]))
    v = v.reshape(a_shape[:-2] + (v.shape[-2:]))
    return u, s, v


def syevj(a, UPLO='L', with_eigen_vector=True):
    """Eigenvalue decomposition of symmetric matrix using cusolverDn<t>syevj().

    Computes eigenvalues ``w`` and (optionally) eigenvectors ``v`` of a complex
    Hermitian or a real symmetric matrix.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
        with_eigen_vector (bool): Indicates whether or not eigenvectors
            are computed.

    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``. For batch input,
            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue
            ``w[k, i]`` of ``a[k]``.
    """
    if not check_availability('syevj'):
        raise RuntimeError('syevj is not available.')

    if UPLO not in ('L', 'U'):
        raise ValueError('UPLO argument must be \'L\' or \'U\'')

    if a.ndim > 2:
        return _syevj_batched(a, UPLO, with_eigen_vector)

    assert a.ndim == 2

    if a.dtype == 'f' or a.dtype == 'e':
        dtype = 'f'
        inp_w_dtype = 'f'
        inp_v_dtype = 'f'
        ret_w_dtype = a.dtype
        ret_v_dtype = a.dtype
    elif a.dtype == 'd':
        dtype = 'd'
        inp_w_dtype = 'd'
        inp_v_dtype = 'd'
        ret_w_dtype = 'd'
        ret_v_dtype = 'd'
    elif a.dtype == 'F':
        dtype = 'F'
        inp_w_dtype = 'f'
        inp_v_dtype = 'F'
        ret_w_dtype = 'f'
        ret_v_dtype = 'F'
    elif a.dtype == 'D':
        dtype = 'D'
        inp_w_dtype = 'd'
        inp_v_dtype = 'D'
        ret_w_dtype = 'd'
        ret_v_dtype = 'D'
    else:
        # NumPy uses float64 when an input is not floating point number.
        dtype = 'd'
        inp_w_dtype = 'd'
        inp_v_dtype = 'd'
        ret_w_dtype = 'd'
        ret_v_dtype = 'd'

    # Note that cuSolver assumes fortran array
    v = a.astype(inp_v_dtype, order='F', copy=True)

    m, lda = a.shape
    w = _cupy.empty(m, inp_w_dtype)
    dev_info = _cupy.empty((), _numpy.int32)
    handle = _device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = _cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = _cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = _cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == 'f':
        buffer_size = _cusolver.ssyevj_bufferSize
        syevj = _cusolver.ssyevj
    elif dtype == 'd':
        buffer_size = _cusolver.dsyevj_bufferSize
        syevj = _cusolver.dsyevj
    elif dtype == 'F':
        buffer_size = _cusolver.cheevj_bufferSize
        syevj = _cusolver.cheevj
    elif dtype == 'D':
        buffer_size = _cusolver.zheevj_bufferSize
        syevj = _cusolver.zheevj
    else:
        raise RuntimeError('Only float and double and cuComplex and '
                           + 'cuDoubleComplex are supported')

    params = _cusolver.createSyevjInfo()
    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr, params)
    work = _cupy.empty(work_size, inp_v_dtype)
    syevj(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr, params)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        syevj, dev_info)

    _cusolver.destroySyevjInfo(params)

    w = w.astype(ret_w_dtype, copy=False)
    if not with_eigen_vector:
        return w
    v = v.astype(ret_v_dtype, copy=False)
    return w, v


def _syevj_batched(a, UPLO, with_eigen_vector):
    if a.dtype == 'f' or a.dtype == 'e':
        dtype = 'f'
        inp_w_dtype = 'f'
        inp_v_dtype = 'f'
        ret_w_dtype = a.dtype
        ret_v_dtype = a.dtype
    elif a.dtype == 'd':
        dtype = 'd'
        inp_w_dtype = 'd'
        inp_v_dtype = 'd'
        ret_w_dtype = 'd'
        ret_v_dtype = 'd'
    elif a.dtype == 'F':
        dtype = 'F'
        inp_w_dtype = 'f'
        inp_v_dtype = 'F'
        ret_w_dtype = 'f'
        ret_v_dtype = 'F'
    elif a.dtype == 'D':
        dtype = 'D'
        inp_w_dtype = 'd'
        inp_v_dtype = 'D'
        ret_w_dtype = 'd'
        ret_v_dtype = 'D'
    else:
        # NumPy uses float64 when an input is not floating point number.
        dtype = 'd'
        inp_w_dtype = 'd'
        inp_v_dtype = 'd'
        ret_w_dtype = 'd'
        ret_v_dtype = 'd'

    *batch_shape, m, lda = a.shape
    batch_size = _numpy.prod(batch_shape)
    a = a.reshape(batch_size, m, lda)
    v = _cupy.array(
        a.swapaxes(-2, -1), order='C', copy=True, dtype=inp_v_dtype)

    w = _cupy.empty((batch_size, m), inp_w_dtype).swapaxes(-2, 1)
    dev_info = _cupy.empty((), _numpy.int32)
    handle = _device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = _cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = _cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = _cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == 'f':
        buffer_size = _cusolver.ssyevjBatched_bufferSize
        syevjBatched = _cusolver.ssyevjBatched
    elif dtype == 'd':
        buffer_size = _cusolver.dsyevjBatched_bufferSize
        syevjBatched = _cusolver.dsyevjBatched
    elif dtype == 'F':
        buffer_size = _cusolver.cheevjBatched_bufferSize
        syevjBatched = _cusolver.cheevjBatched
    elif dtype == 'D':
        buffer_size = _cusolver.zheevjBatched_bufferSize
        syevjBatched = _cusolver.zheevjBatched
    else:
        raise RuntimeError('Only float and double and cuComplex and '
                           + 'cuDoubleComplex are supported')

    params = _cusolver.createSyevjInfo()
    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr, params, batch_size)
    work = _cupy.empty(work_size, inp_v_dtype)
    syevjBatched(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr, params,
        batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        syevjBatched, dev_info)

    _cusolver.destroySyevjInfo(params)

    w = w.astype(ret_w_dtype, copy=False)
    w = w.swapaxes(-2, -1).reshape(*batch_shape, m)
    if not with_eigen_vector:
        return w
    v = v.astype(ret_v_dtype, copy=False)
    v = v.swapaxes(-2, -1).reshape(*batch_shape, m, m)
    return w, v
