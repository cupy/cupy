import numpy

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def _syevd(a, UPLO, with_eigen_vector):
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    # Note that cuSolver uses tranpose of a matrix.
    if not issubclass(a.dtype.type, numpy.inexact):
        # NumPy uses float64 when an input is not floating point number.
        v = a.T.astype('d', copy=True)
        ret_type = numpy.float64
    elif a.dtype == 'e':
        # cuSolver does not have API for float16.
        v = a.T.astype('f', copy=True)
        ret_type = numpy.float16
    else:
        v = a.T.copy()
        ret_type = a.dtype

    dtype = v.dtype
    m, lda = a.shape
    W = cupy.empty(m, dtype)
    dev_info = cupy.empty((), 'i')
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
    else:
        raise RuntimeError('Only float and double are supported')

    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, W.data.ptr)
    work = cupy.empty(work_size, dtype)
    syevd(
        handle, jobz, uplo, m, v.data.ptr, lda,
        W.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)

    return W.astype(ret_type, copy=False), v.T.astype(ret_type, copy=False)


# TODO(okuta): Implement eig


def eigh(a, UPLO='L'):
    """Eigenvalues and eigenvectors of a symmetric matrix.

    This method calculates eigenvalues and eigenvectors of a given
    symmetric matrix.

    .. note::

       Currenlty only 2-D matrix is supported.

    .. note::

       CUDA >=8.0 is required.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        tuple of :class:`~cupy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')
    return _syevd(a, UPLO, True)


# TODO(okuta): Implement eigvals


def eigvalsh(a, UPLO='L'):
    """Calculates eigenvalues of a symmetric matrix.

    This method calculates eigenvalues a given symmetric matrix.
    Note that :func:`cupy.linalg.eigh` calculates both eigenvalues and
    eigenvectors.

    .. note::

       Currenlty only 2-D matrix is supported.

    .. note::

       CUDA >=8.0 is required.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')
    return _syevd(a, UPLO, False)[0]
