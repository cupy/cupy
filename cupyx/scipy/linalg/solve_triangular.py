import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import util


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False):
    """Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        lower (bool): Use only data contained in the lower triangle of `a`.
            Default is to use upper triangle.
        trans ({0, 1, 2, 'N', 'T', 'C'}): Type of system to solve:
            ========  =========
            trans     system
            ========  =========
            0 or 'N'  a x  = b
            1 or 'T'  a^T x = b
            2 or 'C'  a^H x = b
            ========  =========
        unit_diagonal (bool): If True, diagonal elements of `a` are assumed to
            be 1 and will not be referenced.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`numpy.linalg.solve_triangular`
    """

    util._assert_cupy_array(a, b)
    util._assert_nd_squareness(a)

    if not ((a.ndim == b.ndim or a.ndim == b.ndim + 1) and
            a.shape[:-1] == b.shape[:a.ndim - 1]):
        raise ValueError(
            'a must have (..., M, M) shape and b must have (..., M) '
            'or (..., M, K)')

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ())

    a = a.astype(dtype)
    b = b.astype(dtype)

    a = cupy.asfortranarray(a)
    b = cupy.asfortranarray(b)
    dtype = a.dtype
    m, n = (b.size, 1) if b.ndim == 1 else b.shape
    cublas_handle = device.get_cublas_handle()

    if dtype == 'f':
        trsm = cublas.strsm
    else:  # dtype == 'd'
        trsm = cublas.dtrsm

    if lower is True:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if trans == 'N':
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T':
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C':
        trans = cublas.CUBLAS_OP_C

    if unit_diagonal is True:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, uplo,
        trans, diag,
        m, n, 1, a.data.ptr, m, b.data.ptr, m)
    return b
