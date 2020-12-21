import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False):
    """Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        lower (bool): Use only data contained in the lower triangle of ``a``.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T' or 'C'): Type of system to solve:

            - *'0'* or *'N'* -- :math:`a x  = b`
            - *'1'* or *'T'* -- :math:`a^T x = b`
            - *'2'* or *'C'* -- :math:`a^H x = b`

        unit_diagonal (bool): If ``True``, diagonal elements of ``a`` are
            assumed to be 1 and will not be referenced.
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.solve_triangular`
    """

    _util._assert_cupy_array(a, b)

    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix')
    if len(a) != len(b):
        raise ValueError('incompatible dimensions')

    # Cast to float32 or float64
    if a.dtype.char in 'fd':
        dtype = a.dtype
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f')

    a = cupy.array(a, dtype=dtype, order='F', copy=False)
    b = cupy.array(b, dtype=dtype, order='F', copy=(not overwrite_b))

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    m, n = (b.size, 1) if b.ndim == 1 else b.shape
    cublas_handle = device.get_cublas_handle()

    if dtype == 'f':
        trsm = cublas.strsm
    elif dtype == 'd':
        trsm = cublas.dtrsm
    elif dtype == 'F':
        trsm = cublas.ctrsm
    else:  # dtype == 'D'
        trsm = cublas.ztrsm
    one = numpy.array(1, dtype=dtype)

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if trans == 'N':
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T':
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C':
        trans = cublas.CUBLAS_OP_C

    if unit_diagonal:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, uplo,
        trans, diag,
        m, n, one.ctypes.data, a.data.ptr, m, b.data.ptr, m)
    return b
