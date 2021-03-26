from warnings import warn

import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.linalg import _util


def lu_factor(a, overwrite_a=False, check_finite=True):
    """LU decomposition.

    Decompose a given two-dimensional square matrix into ``P * L * U``,
    where ``P`` is a permutation matrix,  ``L`` lower-triangular with
    unit diagonal elements, and ``U`` upper-triangular matrix.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(M, N)``
        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        tuple:
            ``(lu, piv)`` where ``lu`` is a :class:`cupy.ndarray`
            storing ``U`` in its upper triangle, and ``L`` without
            unit diagonal elements in its lower triangle, and ``piv`` is
            a :class:`cupy.ndarray` storing pivot indices representing
            permutation matrix ``P``. For ``0 <= i < min(M,N)``, row
            ``i`` of the matrix was interchanged with row ``piv[i]``

    .. seealso:: :func:`scipy.linalg.lu_factor`
    """
    return _lu_factor(a, overwrite_a, check_finite)


def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    """LU decomposition.

    Decomposes a given two-dimensional matrix into ``P @ L @ U``, where ``P``
    is a permutation matrix, ``L`` is a lower triangular or trapezoidal matrix
    with unit diagonal, and ``U`` is a upper triangular or trapezoidal matrix.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(M, N)``.
        permute_l (bool): If ``True``, perform the multiplication ``P @ L``.
        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        tuple:
            ``(P, L, U)`` if ``permute_l == False``, otherwise ``(PL, U)``.
            ``P`` is a :class:`cupy.ndarray` storing permutation matrix with
            dimension ``(M, M)``. ``L`` is a :class:`cupy.ndarray` storing
            lower triangular or trapezoidal matrix with unit diagonal with
            dimension ``(M, K)`` where ``K = min(M, N)``. ``U`` is a
            :class:`cupy.ndarray` storing upper triangular or trapezoidal
            matrix with dimension ``(K, N)``. ``PL`` is a :class:`cupy.ndarray`
            storing permuted ``L`` matrix with dimension ``(M, K)``.

    .. seealso:: :func:`scipy.linalg.lu`
    """
    lu, piv = _lu_factor(a, overwrite_a, check_finite)

    m, n = lu.shape
    k = min(m, n)
    L, U = _cupy_split_lu(lu)

    if permute_l:
        _cupy_laswp(L, 0, k-1, piv, -1)
        return (L, U)
    else:
        r_dtype = numpy.float32 if lu.dtype.char in 'fF' else numpy.float64
        P = cupy.diag(cupy.ones((m,), dtype=r_dtype))
        _cupy_laswp(P, 0, k-1, piv, -1)
        return (P, L, U)


def _lu_factor(a, overwrite_a=False, check_finite=True):
    a = cupy.asarray(a)
    _util._assert_rank2(a)

    dtype = a.dtype

    if dtype.char == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
    elif dtype.char == 'd':
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
    elif dtype.char == 'F':
        getrf = cusolver.cgetrf
        getrf_bufferSize = cusolver.cgetrf_bufferSize
    elif dtype.char == 'D':
        getrf = cusolver.zgetrf
        getrf_bufferSize = cusolver.zgetrf_bufferSize
    else:
        msg = 'Only float32, float64, complex64 and complex128 are supported.'
        raise NotImplementedError(msg)

    a = a.astype(dtype, order='F', copy=(not overwrite_a))

    if check_finite:
        if a.dtype.kind == 'f' and not cupy.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    m, n = a.shape

    ipiv = cupy.empty((min(m, n),), dtype=numpy.intc)

    buffersize = getrf_bufferSize(cusolver_handle, m, n, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)

    # LU factorization
    getrf(cusolver_handle, m, n, a.data.ptr, m, workspace.data.ptr,
          ipiv.data.ptr, dev_info.data.ptr)

    if dev_info[0] < 0:
        raise ValueError('illegal value in %d-th argument of '
                         'internal getrf (lu_factor)' % -dev_info[0])
    elif dev_info[0] > 0:
        warn('Diagonal number %d is exactly zero. Singular matrix.'
             % dev_info[0], RuntimeWarning, stacklevel=2)

    # cuSolver uses 1-origin while SciPy uses 0-origin
    ipiv -= 1

    return (a, ipiv)


def _cupy_split_lu(LU, order='C'):
    assert LU._f_contiguous
    m, n = LU.shape
    k = min(m, n)
    order = 'F' if order == 'F' else 'C'
    L = cupy.empty((m, k), order=order, dtype=LU.dtype)
    U = cupy.empty((k, n), order=order, dtype=LU.dtype)
    size = m * n
    _kernel_cupy_split_lu(LU, m, n, k, L._c_contiguous, L, U, size=size)
    return (L, U)


_device_get_index = '''
__device__ inline int get_index(int row, int col, int num_rows, int num_cols,
                                bool c_contiguous)
{
    if (c_contiguous) {
        return col + num_cols * row;
    } else {
        return row + num_rows * col;
    }
}
'''

_kernel_cupy_split_lu = cupy.ElementwiseKernel(
    'raw T LU, int32 M, int32 N, int32 K, bool C_CONTIGUOUS',
    'raw T L, raw T U',
    '''
    // LU: shape: (M, N)
    // L: shape: (M, K)
    // U: shape: (K, N)
    const T* ptr_LU = &(LU[0]);
    T* ptr_L = &(L[0]);
    T* ptr_U = &(U[0]);
    int row, col;
    if (C_CONTIGUOUS) {
        row = i / N;
        col = i % N;
    } else {
        row = i % M;
        col = i / M;
    }
    T lu_val = ptr_LU[get_index(row, col, M, N, false)];
    T l_val, u_val;
    if (row > col) {
        l_val = lu_val;
        u_val = static_cast<T>(0);
    } else if (row == col) {
        l_val = static_cast<T>(1);
        u_val = lu_val;
    } else {
        l_val = static_cast<T>(0);
        u_val = lu_val;
    }
    if (col < K) {
        ptr_L[get_index(row, col, M, K, C_CONTIGUOUS)] = l_val;
    }
    if (row < K) {
        ptr_U[get_index(row, col, K, N, C_CONTIGUOUS)] = u_val;
    }
    ''',
    'cupy_split_lu', preamble=_device_get_index
)


def _cupy_laswp(A, k1, k2, ipiv, incx):
    m, n = A.shape
    k = ipiv.shape[0]
    assert 0 <= k1 and k1 <= k2 and k2 < k
    assert A._c_contiguous or A._f_contiguous
    _kernel_cupy_laswp(m, n, k1, k2, ipiv, incx, A._c_contiguous, A, size=n)


_kernel_cupy_laswp = cupy.ElementwiseKernel(
    'int32 M, int32 N, int32 K1, int32 K2, raw I IPIV, int32 INCX, '
    'bool C_CONTIGUOUS',
    'raw T A',
    '''
    // IPIV: 0-based pivot indices. shape: (K,)  (*) K > K2
    // A: shape: (M, N)
    T* ptr_A = &(A[0]);
    if (K1 > K2) return;
    int row_start, row_end, row_inc;
    if (INCX > 0) {
        row_start = K1; row_end = K2; row_inc = 1;
    } else if (INCX < 0) {
        row_start = K2; row_end = K1; row_inc = -1;
    } else {
        return;
    }
    int col = i;
    int row1 = row_start;
    while (1) {
        int row2 = IPIV[row1];
        if (row1 != row2) {
            int idx1 = get_index(row1, col, M, N, C_CONTIGUOUS);
            int idx2 = get_index(row2, col, M, N, C_CONTIGUOUS);
            T tmp       = ptr_A[idx1];
            ptr_A[idx1] = ptr_A[idx2];
            ptr_A[idx2] = tmp;
        }
        if (row1 == row_end) break;
        row1 += row_inc;
    }
    ''',
    'cupy_laswp', preamble=_device_get_index
)


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, ``a * x = b``, given the LU factorization of ``a``

    Args:
        lu_and_piv (tuple): LU factorization of matrix ``a`` (``(M, M)``)
            together with pivot indices.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        trans ({0, 1, 2}): Type of system to solve:

            ========  =========
            trans     system
            ========  =========
            0         a x  = b
            1         a^T x = b
            2         a^H x = b
            ========  =========
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.lu_solve`
    """

    (lu, ipiv) = lu_and_piv

    _util._assert_cupy_array(lu)
    _util._assert_rank2(lu)
    _util._assert_nd_squareness(lu)

    m = lu.shape[0]
    if m != b.shape[0]:
        raise ValueError('incompatible dimensions.')

    dtype = lu.dtype
    if dtype.char == 'f':
        getrs = cusolver.sgetrs
    elif dtype.char == 'd':
        getrs = cusolver.dgetrs
    elif dtype.char == 'F':
        getrs = cusolver.cgetrs
    elif dtype.char == 'D':
        getrs = cusolver.zgetrs
    else:
        msg = 'Only float32, float64, complex64 and complex128 are supported.'
        raise NotImplementedError(msg)

    if trans == 0:
        trans = cublas.CUBLAS_OP_N
    elif trans == 1:
        trans = cublas.CUBLAS_OP_T
    elif trans == 2:
        trans = cublas.CUBLAS_OP_C
    else:
        raise ValueError('unknown trans')

    lu = lu.astype(dtype, order='F', copy=False)
    ipiv = ipiv.astype(ipiv.dtype, order='F', copy=True)
    # cuSolver uses 1-origin while SciPy uses 0-origin
    ipiv += 1
    b = b.astype(dtype, order='F', copy=(not overwrite_b))

    if check_finite:
        if lu.dtype.kind == 'f' and not cupy.isfinite(lu).all():
            raise ValueError(
                'array must not contain infs or NaNs.\n'
                'Note that when a singular matrix is given, unlike '
                'scipy.linalg.lu_factor, cupyx.scipy.linalg.lu_factor '
                'returns an array containing NaN.')
        if b.dtype.kind == 'f' and not cupy.isfinite(b).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    n = 1 if b.ndim == 1 else b.shape[1]
    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    # solve for the inverse
    getrs(cusolver_handle,
          trans,
          m, n, lu.data.ptr, m, ipiv.data.ptr, b.data.ptr,
          m, dev_info.data.ptr)

    if dev_info[0] < 0:
        raise ValueError('illegal value in %d-th argument of '
                         'internal getrs (lu_solve)' % -dev_info[0])

    return b
