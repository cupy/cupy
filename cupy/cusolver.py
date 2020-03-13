import numpy

import cupy
from cupy.cuda import cusolver
from cupy.cuda import runtime
from cupy.cuda import device


_available_cuda_version = {
    'gesvdj': (9000, None),
    'gesvda': (10010, None),
}


def check_availability(name):
    if name not in _available_cuda_version:
        msg = 'No available version information specified for {}'.name
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
