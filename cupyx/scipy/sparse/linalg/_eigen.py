import numpy
import cupy

from cupy import cublas
from cupy import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import csr
from cupyx.scipy.sparse.linalg import _interface


def eigsh(a, k=6, *, which='LM', ncv=None, maxiter=None, tol=0,
          return_eigenvectors=True):
    """Finds ``k`` eigenvalues and eigenvectors of the real symmetric matrix.

    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues
    with corresponding eigenvectors ``x``.

    Args:
        a (ndarray, spmatrix or LinearOperator): A symmetric square matrix with
            dimension ``(n, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of eigenvalues and eigenvectors to compute. Must be
            ``1 <= k < n``.
        which (str): 'LM' or 'LA'. 'LM': finds ``k`` largest (in magnitude)
            eigenvalues. 'LA': finds ``k`` largest (algebraic) eigenvalues.
        ncv (int): The number of Lanczos vectors generated. Must be
            ``k + 1 < ncv < n``. If ``None``, default value is used.
        maxiter (int): Maximum number of Lanczos update iterations.
            If ``None``, default value is used.
        tol (float): Tolerance for residuals ``||Ax - wx||``. If ``0``, machine
            precision is used.
        return_eigenvectors (bool): If ``True``, returns eigenvectors in
            addition to eigenvalues.

    Returns:
        tuple:
            If ``return_eigenvectors is True``, it returns ``w`` and ``x``
            where ``w`` is eigenvalues and ``x`` is eigenvectors. Otherwise,
            it returns only ``w``.

    .. seealso:: :func:`scipy.sparse.linalg.eigsh`

    .. note::
        This function uses the thick-restart Lanczos methods
        (https://sdm.lbl.gov/~kewu/ps/trlan.html).

    """
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= n:
        raise ValueError('k must be smaller than n (actual: {})'.format(k))
    if which not in ('LM', 'LA'):
        raise ValueError('which must be \'LM\' or \'LA\' (actual: {})'
                         ''.format(which))
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = numpy.finfo(a.dtype).eps

    alpha = cupy.zeros((ncv,), dtype=a.dtype)
    beta = cupy.zeros((ncv,), dtype=a.dtype.char.lower())
    V = cupy.empty((ncv, n), dtype=a.dtype)

    # Set initial vector
    u = cupy.random.random((n,)).astype(a.dtype)
    V[0] = u / cublas.nrm2(u)

    # Choose Lanczos implementation, unconditionally use 'fast' for now
    upadte_impl = 'fast'
    if upadte_impl == 'fast':
        lanczos = _lanczos_fast(a, n, ncv)
    else:
        lanczos = _lanczos_asis

    # Lanczos iteration
    lanczos(a, V, u, alpha, beta, 0, ncv)

    iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    # Compute residual
    beta_k = beta[-1] * s[-1, :]
    res = cublas.nrm2(beta_k)

    while res > tol and iter < maxiter:
        # Setup for thick-restart
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T

        u -= u.T @ V[:k].conj().T @ V[:k]
        V[k] = u / cublas.nrm2(u)

        u[...] = a @ V[k]
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        V[k+1] = u / beta[k]

        # Lanczos iteration
        lanczos(a, V, u, alpha, beta, k + 1, ncv)

        iter += ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s

        # Compute residual
        beta_k = beta[-1] * s[-1, :]
        res = cublas.nrm2(beta_k)

    if return_eigenvectors:
        idx = cupy.argsort(w)
        return w[idx], x[:, idx]
    else:
        return cupy.sort(w)


def _lanczos_asis(a, V, u, alpha, beta, i_start, i_end):
    for i in range(i_start, i_end):
        u[...] = a @ V[i]
        cublas.dotc(V[i], u, out=alpha[i])
        u -= u.T @ V[:i+1].conj().T @ V[:i+1]
        cublas.nrm2(u, out=beta[i])
        if i >= i_end - 1:
            break
        V[i+1] = u / beta[i]


def _lanczos_fast(A, n, ncv):
    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
    if A.dtype.char == 'f':
        dotc = _cublas.sdot
        nrm2 = _cublas.snrm2
        gemm = _cublas.sgemm
    elif A.dtype.char == 'd':
        dotc = _cublas.ddot
        nrm2 = _cublas.dnrm2
        gemm = _cublas.dgemm
    elif A.dtype.char == 'F':
        dotc = _cublas.cdotc
        nrm2 = _cublas.scnrm2
        gemm = _cublas.cgemm
    elif A.dtype.char == 'D':
        dotc = _cublas.zdotc
        nrm2 = _cublas.dznrm2
        gemm = _cublas.zgemm
    else:
        raise TypeError('invalid dtype ({})'.format(A.dtype))

    cusparse_handle = None
    if csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT

    v = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)

    outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end):
        assert A is outer_A

        # Get ready for spmv if enabled
        if cusparse_handle is not None:
            # Note: I would like to reuse descriptors and working buffer
            # on the next update, but I gave it up because it sometimes
            # caused illegal memory access error.
            spmv_desc_A = cusparse.SpMatDescriptor.create(A)
            spmv_desc_v = cusparse.DnVecDescriptor.create(v)
            spmv_desc_u = cusparse.DnVecDescriptor.create(u)
            buff_size = _cusparse.spMV_bufferSize(
                cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data,
                spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)

        v[...] = V[i_start]
        for i in range(i_start, i_end):
            # Matrix-vector multiplication
            if cusparse_handle is None:
                u[...] = A @ v
            else:
                _cusparse.spMV(
                    cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data,
                    spmv_desc_A.desc, spmv_desc_v.desc,
                    spmv_beta.ctypes.data, spmv_desc_u.desc,
                    spmv_cuda_dtype, spmv_alg, spmv_buff.data.ptr)

            # Call dotc
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1,
                     alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)

            # Orthogonalize
            gemm(cublas_handle, _cublas.CUBLAS_OP_C, _cublas.CUBLAS_OP_N,
                 1, i + 1, n,
                 one.ctypes.data, u.data.ptr, n, V.data.ptr, n,
                 zero.ctypes.data, uu.data.ptr, 1)
            gemm(cublas_handle, _cublas.CUBLAS_OP_N, _cublas.CUBLAS_OP_C,
                 n, 1, i + 1,
                 mone.ctypes.data, V.data.ptr, n, uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, n)

            # Call nrm2
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1,
                     beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)

            # Break here as the normalization below touches V[i+1]
            if i >= i_end - 1:
                break

            # Normalize
            _kernel_normalize(u, beta, i, n, v, V)

    return aux


_kernel_normalize = cupy.ElementwiseKernel(
    'T u, raw S beta, int32 j, int32 n', 'T v, raw T V',
    'v = u / beta[j]; V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)


def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    # Note: This is done on the CPU, because there is an issue in
    # cupy.linalg.eigh with CUDA 9.2, which can return NaNs. It will has little
    # impact on performance, since the matrix size processed here is not large.
    alpha = cupy.asnumpy(alpha)
    beta = cupy.asnumpy(beta)
    t = numpy.diag(alpha)
    t = t + numpy.diag(beta[:-1], k=1)
    t = t + numpy.diag(beta[:-1], k=-1)
    if beta_k is not None:
        beta_k = cupy.asnumpy(beta_k)
        t[k, :k] = beta_k
        t[:k, k] = beta_k
    w, s = numpy.linalg.eigh(t)

    # Pick-up k ritz-values and ritz-vectors
    if which == 'LA':
        idx = numpy.argsort(w)
    elif which == 'LM':
        idx = numpy.argsort(numpy.absolute(w))
    wk = w[idx[-k:]]
    sk = s[:, idx[-k:]]
    return cupy.array(wk), cupy.array(sk)


def svds(a, k=6, *, ncv=None, tol=0, which='LM', maxiter=None,
         return_singular_vectors=True):
    """Finds the largest ``k`` singular values/vectors for a sparse matrix.

    Args:
        a (ndarray, spmatrix or LinearOperator): A real or complex array with
            dimension ``(m, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of singular values/vectors to compute. Must be
            ``1 <= k < min(m, n)``.
        ncv (int): The number of Lanczos vectors generated. Must be
            ``k + 1 < ncv < min(m, n)``. If ``None``, default value is used.
        tol (float): Tolerance for singular values. If ``0``, machine precision
            is used.
        which (str): Only 'LM' is supported. 'LM': finds ``k`` largest singular
            values.
        maxiter (int): Maximum number of Lanczos update iterations.
            If ``None``, default value is used.
        return_singular_vectors (bool): If ``True``, returns singular vectors
            in addition to singular values.

    Returns:
        tuple:
            If ``return_singular_vectors`` is ``True``, it returns ``u``, ``s``
            and ``vt`` where ``u`` is left singular vectors, ``s`` is singular
            values and ``vt`` is right singular vectors. Otherwise, it returns
            only ``s``.

    .. seealso:: :func:`scipy.sparse.linalg.svds`

    .. note::
        This is a naive implementation using cupyx.scipy.sparse.linalg.eigsh as
        an eigensolver on ``a.H @ a`` or ``a @ a.H``.

    """
    if a.ndim != 2:
        raise ValueError('expected 2D (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    m, n = a.shape
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= min(m, n):
        raise ValueError('k must be smaller than min(m, n) (actual: {})'
                         ''.format(k))

    a = _interface.aslinearoperator(a)
    if m >= n:
        aH, a = a.H, a
    else:
        aH, a = a, a.H

    if return_singular_vectors:
        w, x = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter,
                     tol=tol, return_eigenvectors=True)
    else:
        w = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol,
                  return_eigenvectors=False)

    w = cupy.maximum(w, 0)
    t = w.dtype.char.lower()
    factor = {'f': 1e3, 'd': 1e6}
    cond = factor[t] * numpy.finfo(t).eps
    cutoff = cond * cupy.max(w)
    above_cutoff = (w > cutoff)
    n_large = above_cutoff.sum()
    s = cupy.zeros_like(w)
    s[:n_large] = cupy.sqrt(w[above_cutoff])
    if not return_singular_vectors:
        return s

    x = x[:, above_cutoff]
    if m >= n:
        v = x
        u = a @ v / s[:n_large]
    else:
        u = x
        v = a @ u / s[:n_large]
    u = _augmented_orthnormal_cols(u, k - n_large)
    v = _augmented_orthnormal_cols(v, k - n_large)

    return u, s, v.conj().T


def _augmented_orthnormal_cols(x, n_aug):
    if n_aug <= 0:
        return x
    m, n = x.shape
    y = cupy.empty((m, n + n_aug), dtype=x.dtype)
    y[:, :n] = x
    for i in range(n, n + n_aug):
        v = cupy.random.random((m, )).astype(x.dtype)
        v -= v @ y[:, :i].conj() @ y[:, :i].T
        y[:, i] = v / cupy.linalg.norm(v)
    return y
