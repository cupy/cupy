import numpy

import cupy
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy import util


_available_cuda_version = {
    'gesvdj': (9000, None),
    'gesvda': (10010, None),
    'potrfBatched': (9010, None),
    'potrsBatched': (9010, None),
    'syevj': (9000, None),
}


@util.memoize()
def check_availability(name):
    if name not in _available_cuda_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = _available_cuda_version[name]
    cuda_version = runtime.runtimeGetVersion()
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
        helper = cusolver.sgesvdj_bufferSize
        solver = cusolver.sgesvdj
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = cusolver.dgesvdj_bufferSize
        solver = cusolver.dgesvdj
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = cusolver.cgesvdj_bufferSize
        solver = cusolver.cgesvdj
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = cusolver.zgesvdj_bufferSize
        solver = cusolver.zgesvdj
        s_dtype = 'd'
    else:
        raise TypeError

    handle = device.get_cusolver_handle()
    m, n = a.shape
    a = cupy.array(a, order='F', copy=not overwrite_a)
    lda = m
    mn = min(m, n)
    s = cupy.empty(mn, dtype=s_dtype)
    ldu = m
    ldv = n
    if compute_uv:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
        full_matrices = False
    if full_matrices:
        econ = 0
        u = cupy.empty((ldu, m), dtype=a.dtype, order='F')
        v = cupy.empty((ldv, n), dtype=a.dtype, order='F')
    else:
        econ = 1
        u = cupy.empty((ldu, mn), dtype=a.dtype, order='F')
        v = cupy.empty((ldv, mn), dtype=a.dtype, order='F')
    params = cusolver.createGesvdjInfo()
    lwork = helper(handle, jobz, econ, m, n, a.data.ptr, lda, s.data.ptr,
                   u.data.ptr, ldu, v.data.ptr, ldv, params)
    work = cupy.empty(lwork, dtype=a.dtype)
    info = cupy.empty(1, dtype=numpy.int32)
    solver(handle, jobz, econ, m, n, a.data.ptr, lda, s.data.ptr,
           u.data.ptr, ldu, v.data.ptr, ldv, work.data.ptr, lwork,
           info.data.ptr, params)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvdj, info)

    cusolver.destroyGesvdjInfo(params)
    if compute_uv:
        return u, s, v
    else:
        return s


def _gesvdj_batched(a, full_matrices, compute_uv, overwrite_a):
    if a.dtype == 'f':
        helper = cusolver.sgesvdjBatched_bufferSize
        solver = cusolver.sgesvdjBatched
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = cusolver.dgesvdjBatched_bufferSize
        solver = cusolver.dgesvdjBatched
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = cusolver.cgesvdjBatched_bufferSize
        solver = cusolver.cgesvdjBatched
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = cusolver.zgesvdjBatched_bufferSize
        solver = cusolver.zgesvdjBatched
        s_dtype = 'd'
    else:
        raise TypeError

    handle = device.get_cusolver_handle()
    batch_size, m, n = a.shape
    a = cupy.array(a.swapaxes(-2, -1), order='C', copy=not overwrite_a)
    lda = m
    mn = min(m, n)
    s = cupy.empty((batch_size, mn), dtype=s_dtype)
    ldu = m
    ldv = n
    if compute_uv:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
        # if not batched, `full_matrices = False` could speedup.

    u = cupy.empty((batch_size, m, ldu), dtype=a.dtype).swapaxes(-2, -1)
    v = cupy.empty((batch_size, n, ldv), dtype=a.dtype).swapaxes(-2, -1)
    params = cusolver.createGesvdjInfo()
    lwork = helper(handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
                   u.data.ptr, ldu, v.data.ptr, ldv, params, batch_size)
    work = cupy.empty(lwork, dtype=a.dtype)
    info = cupy.empty(1, dtype=numpy.int32)
    solver(handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
           u.data.ptr, ldu, v.data.ptr, ldv, work.data.ptr, lwork,
           info.data.ptr, params, batch_size)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvdj, info)

    cusolver.destroyGesvdjInfo(params)
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
        helper = cusolver.sgesvdaStridedBatched_bufferSize
        solver = cusolver.sgesvdaStridedBatched
        s_dtype = 'f'
    elif a.dtype == 'd':
        helper = cusolver.dgesvdaStridedBatched_bufferSize
        solver = cusolver.dgesvdaStridedBatched
        s_dtype = 'd'
    elif a.dtype == 'F':
        helper = cusolver.cgesvdaStridedBatched_bufferSize
        solver = cusolver.cgesvdaStridedBatched
        s_dtype = 'f'
    elif a.dtype == 'D':
        helper = cusolver.zgesvdaStridedBatched_bufferSize
        solver = cusolver.zgesvdaStridedBatched
        s_dtype = 'd'
    else:
        raise TypeError

    handle = device.get_cusolver_handle()
    if compute_uv:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    rank = min(m, n)
    if a_ndim == 2:
        batch_size = 1
    else:
        batch_size = numpy.array(a_shape[:-2]).prod().item()
    a = a.reshape((batch_size, m, n))
    a = cupy.ascontiguousarray(a.transpose(0, 2, 1))
    lda = m
    stride_a = lda * n
    s = cupy.empty((batch_size, rank), dtype=s_dtype)
    stride_s = rank
    ldu = m
    ldv = n
    u = cupy.empty((batch_size, rank, ldu), dtype=a.dtype, order='C')
    v = cupy.empty((batch_size, rank, ldv), dtype=a.dtype, order='C')
    stride_u = rank * ldu
    stride_v = rank * ldv
    lwork = helper(handle, jobz, rank, m, n, a.data.ptr, lda, stride_a,
                   s.data.ptr, stride_s, u.data.ptr, ldu, stride_u,
                   v.data.ptr, ldv, stride_v, batch_size)
    work = cupy.empty((lwork,), dtype=a.dtype)
    info = cupy.empty((batch_size,), dtype=numpy.int32)
    r_norm = numpy.empty((batch_size,), dtype=numpy.float64)
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
    w = cupy.empty(m, inp_w_dtype)
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == 'f':
        buffer_size = cusolver.ssyevj_bufferSize
        syevj = cusolver.ssyevj
    elif dtype == 'd':
        buffer_size = cusolver.dsyevj_bufferSize
        syevj = cusolver.dsyevj
    elif dtype == 'F':
        buffer_size = cusolver.cheevj_bufferSize
        syevj = cusolver.cheevj
    elif dtype == 'D':
        buffer_size = cusolver.zheevj_bufferSize
        syevj = cusolver.zheevj
    else:
        raise RuntimeError('Only float and double and cuComplex and '
                           + 'cuDoubleComplex are supported')

    params = cusolver.createSyevjInfo()
    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr, params)
    work = cupy.empty(work_size, inp_v_dtype)
    syevj(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr, params)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        syevj, dev_info)

    cusolver.destroySyevjInfo(params)

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
    batch_size = numpy.prod(batch_shape)
    a = a.reshape(batch_size, m, lda)
    v = cupy.array(a.swapaxes(-2, -1), order='C', copy=True, dtype=inp_v_dtype)

    w = cupy.empty((batch_size, m), inp_w_dtype).swapaxes(-2, 1)
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == 'f':
        buffer_size = cusolver.ssyevjBatched_bufferSize
        syevjBatched = cusolver.ssyevjBatched
    elif dtype == 'd':
        buffer_size = cusolver.dsyevjBatched_bufferSize
        syevjBatched = cusolver.dsyevjBatched
    elif dtype == 'F':
        buffer_size = cusolver.cheevjBatched_bufferSize
        syevjBatched = cusolver.cheevjBatched
    elif dtype == 'D':
        buffer_size = cusolver.zheevjBatched_bufferSize
        syevjBatched = cusolver.zheevjBatched
    else:
        raise RuntimeError('Only float and double and cuComplex and '
                           + 'cuDoubleComplex are supported')

    params = cusolver.createSyevjInfo()
    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr, params, batch_size)
    work = cupy.empty(work_size, inp_v_dtype)
    syevjBatched(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr, params,
        batch_size)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        syevjBatched, dev_info)

    cusolver.destroySyevjInfo(params)

    w = w.astype(ret_w_dtype, copy=False)
    w = w.swapaxes(-2, -1).reshape(*batch_shape, m)
    if not with_eigen_vector:
        return w
    v = v.astype(ret_v_dtype, copy=False)
    v = v.swapaxes(-2, -1).reshape(*batch_shape, m, m)
    return w, v
