import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device


def _syevd(a, UPLO, with_eigen_vector):
    if UPLO not in ('L', 'U'):
        raise ValueError('UPLO argument must be \'L\' or \'U\'')

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
    work = cupy.empty(work_size, inp_v_dtype)
    syevd(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        syevd, dev_info)

    return w.astype(ret_w_dtype, copy=False), v.astype(ret_v_dtype, copy=False)


# TODO(okuta): Implement eig


def eigh(a, UPLO='L'):
    """Eigenvalues and eigenvectors of a symmetric matrix.

    This method calculates eigenvalues and eigenvectors of a given
    symmetric matrix.

    .. note::

       Currently only 2-D matrix is supported.

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

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    return _syevd(a, UPLO, True)


# TODO(okuta): Implement eigvals


def eigvalsh(a, UPLO='L'):
    """Calculates eigenvalues of a symmetric matrix.

    This method calculates eigenvalues a given symmetric matrix.
    Note that :func:`cupy.linalg.eigh` calculates both eigenvalues and
    eigenvectors.

    .. note::

       Currenlty only 2-D matrix is supported.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        cupy.ndarray:
            Returns eigenvalues as a vector.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    return _syevd(a, UPLO, False)[0]
