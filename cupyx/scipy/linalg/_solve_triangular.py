import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray


@_uarray.implements('solve_triangular')
def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False):
    """Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(..., M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(..., M,)`` or
            ``(..., M, N)``.
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
            The matrix with dimension ``(..., M,)`` or ``(..., M, N)``.

    .. note::
       Unlike the SciPy counterpart, the CuPy implementation supports batches
       of matrices.
    .. seealso:: :func:`scipy.linalg.solve_triangular`
    """

    _util._assert_cupy_array(a, b)

    if a.ndim == 2:
        if a.shape[0] != a.shape[1]:
            raise ValueError('expected square matrix')
        if len(a) != len(b):
            raise ValueError('incompatible dimensions')
        batch_count = 0
    elif a.ndim > 2:
        if a.shape[-1] != a.shape[-2]:
            raise ValueError('expected a batch of square matrices')
        if a.shape[:-2] != b.shape[:a.ndim - 2]:
            raise ValueError('incompatible batch count')
        if b.ndim < a.ndim - 1 or a.shape[-2] != b.shape[a.ndim - 2]:
            raise ValueError('incompatible dimensions')
        batch_count = numpy.prod(a.shape[:-2])
    else:
        raise ValueError(
            'expected one square matrix or a batch of square matrices')

    # Cast to float32 or float64
    if a.dtype.char in 'fd':
        dtype = a.dtype
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f')

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    if batch_count:
        m, n = b.shape[-2:] if b.ndim == a.ndim else (b.shape[-1], 1)

        a_new_shape = (batch_count, m, m)
        b_shape = b.shape
        b_data_ptr = b.data.ptr
        # trsm receives Fortran array, but we want zero copy
        if trans == 'N' or trans == cublas.CUBLAS_OP_N:
            # normal Fortran upper == transpose C lower
            trans = cublas.CUBLAS_OP_T
            lower = not lower
            a = cupy.ascontiguousarray(a.reshape(*a_new_shape), dtype=dtype)
        elif trans == 'T' or trans == cublas.CUBLAS_OP_T:
            # transpose Fortran upper == normal C lower
            trans = cublas.CUBLAS_OP_N
            lower = not lower
            a = cupy.ascontiguousarray(a.reshape(*a_new_shape), dtype=dtype)
        elif trans == 'C' or trans == cublas.CUBLAS_OP_C:
            if dtype == 'f' or dtype == 'd':
                # real numbers
                # Hermitian Fortran upper == transpose Fortran upper
                #                         == normal C lower
                trans = cublas.CUBLAS_OP_N
                lower = not lower
                a = cupy.ascontiguousarray(a.reshape(*a_new_shape),
                                           dtype=dtype)
            else:
                # complex numbers
                trans = cublas.CUBLAS_OP_C
                a = cupy.ascontiguousarray(
                    a.reshape(*a_new_shape).transpose(0, 2, 1), dtype=dtype)
        else:  # know nothing about `trans`, just convert C to Fortran
            a = cupy.ascontiguousarray(
                a.reshape(*a_new_shape).transpose(0, 2, 1), dtype=dtype)
        b = cupy.ascontiguousarray(
            b.reshape(batch_count, m, n).transpose(0, 2, 1), dtype=dtype)
        if b.data.ptr == b_data_ptr and not overwrite_b:
            b = b.copy()

        start = a.data.ptr
        step = m * m * a.itemsize
        stop = start + step * batch_count
        a_array = cupy.arange(start, stop, step, dtype=cupy.uintp)

        start = b.data.ptr
        step = m * n * b.itemsize
        stop = start + step * batch_count
        b_array = cupy.arange(start, stop, step, dtype=cupy.uintp)
    else:
        a = cupy.array(a, dtype=dtype, order='F', copy=False)
        b = cupy.array(b, dtype=dtype, order='F', copy=(not overwrite_b))

        m, n = (b.size, 1) if b.ndim == 1 else b.shape

        if trans == 'N':
            trans = cublas.CUBLAS_OP_N
        elif trans == 'T':
            trans = cublas.CUBLAS_OP_T
        elif trans == 'C':
            trans = cublas.CUBLAS_OP_C

    cublas_handle = device.get_cublas_handle()
    one = numpy.array(1, dtype=dtype)

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if unit_diagonal:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    if batch_count:
        if dtype == 'f':
            trsm = cublas.strsmBatched
        elif dtype == 'd':
            trsm = cublas.dtrsmBatched
        elif dtype == 'F':
            trsm = cublas.ctrsmBatched
        else:  # dtype == 'D'
            trsm = cublas.ztrsmBatched
        trsm(
            cublas_handle, cublas.CUBLAS_SIDE_LEFT, uplo,
            trans, diag,
            m, n, one.ctypes.data, a_array.data.ptr, m,
            b_array.data.ptr, m, batch_count)
        return b.transpose(0, 2, 1).reshape(b_shape)
    else:
        if dtype == 'f':
            trsm = cublas.strsm
        elif dtype == 'd':
            trsm = cublas.dtrsm
        elif dtype == 'F':
            trsm = cublas.ctrsm
        else:  # dtype == 'D'
            trsm = cublas.ztrsm
        trsm(
            cublas_handle, cublas.CUBLAS_SIDE_LEFT, uplo,
            trans, diag,
            m, n, one.ctypes.data, a.data.ptr, m, b.data.ptr, m)
        return b
