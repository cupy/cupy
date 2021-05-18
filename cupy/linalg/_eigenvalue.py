import numpy

import cupy
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import _util


def _syevd(a, UPLO, with_eigen_vector):
    if UPLO not in ('L', 'U'):
        raise ValueError('UPLO argument must be \'L\' or \'U\'')

    # reject_float16=False for backward compatibility
    dtype, v_dtype = _util.linalg_common_type(a, reject_float16=False)
    real_dtype = dtype.char.lower()
    w_dtype = v_dtype.char.lower()

    # Note that cuSolver assumes fortran array
    v = a.astype(dtype, order='F', copy=True)

    m, lda = a.shape
    w = cupy.empty(m, real_dtype)
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
        buffer_size = cupy.cuda.cusolver.ssyevd_bufferSize
        syevd = cupy.cuda.cusolver.ssyevd
    elif dtype == 'd':
        buffer_size = cupy.cuda.cusolver.dsyevd_bufferSize
        syevd = cupy.cuda.cusolver.dsyevd
    elif dtype == 'F':
        buffer_size = cupy.cuda.cusolver.cheevd_bufferSize
        syevd = cupy.cuda.cusolver.cheevd
    elif dtype == 'D':
        buffer_size = cupy.cuda.cusolver.zheevd_bufferSize
        syevd = cupy.cuda.cusolver.zheevd
    else:
        raise RuntimeError('Only float and double and cuComplex and '
                           + 'cuDoubleComplex are supported')

    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr)
    work = cupy.empty(work_size, dtype)
    syevd(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        syevd, dev_info)

    return w.astype(w_dtype, copy=False), v.astype(v_dtype, copy=False)


# TODO(okuta): Implement eig


def eigh(a, UPLO='L'):
    """Eigenvalues and eigenvectors of a symmetric matrix.

    This method calculates eigenvalues and eigenvectors of a given
    symmetric matrix.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``. For batch input,
            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue
            ``w[k, i]`` of ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    if a.ndim < 2:
        raise ValueError('Array must be at least two-dimensional')

    m, n = a.shape[-2:]
    if m != n:
        raise ValueError('Last 2 dimensions of the array must be square')

    if a.ndim > 2:
        return cupy.cusolver.syevj(a, UPLO, True)
    else:
        return _syevd(a, UPLO, True)


# TODO(okuta): Implement eigvals


def eigvalsh(a, UPLO='L'):
    """Calculates eigenvalues of a symmetric matrix.

    This method calculates eigenvalues a given symmetric matrix.
    Note that :func:`cupy.linalg.eigh` calculates both eigenvalues and
    eigenvectors.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector ``w``. For batch input,
            ``w[k]`` is a vector of eigenvalues of matrix ``a[k]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    if a.ndim < 2:
        raise ValueError('Array must be at least two-dimensional')

    _util._assert_nd_squareness(a)

    if a.ndim > 2:
        return cupy.cusolver.syevj(a, UPLO, False)
    else:
        return _syevd(a, UPLO, False)[0]
