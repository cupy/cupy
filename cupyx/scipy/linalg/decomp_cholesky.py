import numpy

import cupy
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import util


__all__ = ['cholesky', 'cho_factor', 'cho_solve']


def _cholesky(a, lower=False, overwrite_a=False, clean=True,
              check_finite=True):
    """Common code for cholesky() and cho_factor()."""

    util._assert_cupy_array(a)

    # Dimension check
    if a.ndim != 2:
        raise ValueError('Input array needs to be 2D but received '
                         'a {}d-array.'.format(a.ndim))

    # Squareness check
    if a.shape[0] != a.shape[1]:
        raise ValueError('Input array is expected to be square but has '
                         'the shape: {}.'.format(a.shape))

    if check_finite and not cupy.isfinite(a).all():
        raise ValueError(
            "array must not contain infs or NaNs")

    # Quick return for square empty array
    if a.size == 0:
        return a.copy(), lower

    dtype = numpy.promote_types(a.dtype.char, 'f')

    x = a.astype(dtype, order='C', copy=not overwrite_a)
    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
    elif dtype == 'd':
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    elif dtype == 'F':
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
    else:  # dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER

    buffersize = potrf_bufferSize(
        handle, uplo, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    potrf(
        handle, uplo, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    if clean:
        util._tril(x, k=0) if lower else util._triu(x, k=0)
    return x, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
                         check_finite=check_finite)
    return c


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False,
                         check_finite=check_finite)
    return c, lower


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.
    Args:
        c (cupy.ndarray):
            The matrix with dimension (M, M)
            Cholesky factorization of a, as given by cupy.linalg.cholesky
        b (cupy.ndarray):
            The matrix with dimension (M, N) or (M)
            Right-hand side
    Returns:
        cupy.ndarray:
            The matrix with dimension (M, N) or (M)
    """
    (c, lower) = c_and_lower
    util._assert_cupy_array(c)
    util._assert_cupy_array(b)

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("The factored matrix c is not square.")
    if c.shape[1] != b.shape[0]:
        raise ValueError("incompatible dimensions.")

    if check_finite:
        if not cupy.isfinite(c).all() or not cupy.isfinite(b).all():
            raise ValueError(
                "array must not contain infs or NaNs")

    dtype = numpy.promote_types(c.dtype.char, 'f')

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

    if c.flags.f_contiguous:
        c = c.astype(dtype, order='C')

    x = b.astype(dtype, order='F', copy=not overwrite_b)
    b_shape = (b.shape[0], 1) if b.ndim == 1 else b.shape

    uplo = cublas.CUBLAS_FILL_MODE_UPPER
    handle = device.get_cusolver_handle()
    potrs = getattr(cusolver, t + 'potrs')

    n = len(c)
    dev_info = cupy.empty(1, dtype=numpy.int32)
    potrs(handle, uplo, n, b_shape[1], c.data.ptr, n,
          x.data.ptr, n, dev_info.data.ptr)

    return x
