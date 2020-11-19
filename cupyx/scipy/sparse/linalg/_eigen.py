import numpy
import cupy

from cupy import cublas
from cupy import cusparse
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import csr


def eigsh(a, k=6, *, which='LM', ncv=None, maxiter=None, tol=0,
          return_eigenvectors=True):
    """Finds ``k`` eigenvalues and eigenvectors of the real symmetric matrix.

    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues
    with corresponding eigenvectors ``x``.

    Args:
        a (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): A symmetric square
            matrix with dimension ``(n, n)``.
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

    alpha = cupy.zeros((ncv, ), dtype=a.dtype)
    beta = cupy.zeros((ncv, ), dtype=a.dtype.char.lower())
    V = cupy.empty((ncv, n), dtype=a.dtype)
    lanczos = _EigshLanczos(a, V, alpha, beta, update_impl='fast')

    # Set initial vector
    u = cupy.random.random((n, )).astype(a.dtype)
    V[0] = u / cublas.nrm2(u)

    # Lanczos iteration
    u = lanczos.update(0, ncv)
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

        u = a @ V[k]
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        V[k+1] = u / beta[k]

        # Lanczos iteration
        u = lanczos.update(k+1, ncv)
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


class _EigshLanczos():

    def __init__(self, A, V, alpha, beta, update_impl='fast'):
        assert A.ndim == V.ndim == 2
        assert alpha.ndim == beta.ndim == 1
        assert A.dtype == V.dtype == alpha.dtype
        assert A.dtype.char.lower() == beta.dtype.char
        assert A.shape[0] == A.shape[1] == V.shape[1]
        assert V.shape[0] == alpha.shape[0] == beta.shape[0]

        self.A = A
        self.V = V
        self.alpha = alpha
        self.beta = beta
        self.n = V.shape[1]
        self.ncv = V.shape[0]
        self.update_impl = update_impl
        if self.update_impl != 'fast':
            return

        self.cublas_handle = device.get_cublas_handle()
        self.cublas_pointer_mode = _cublas.getPointerMode(self.cublas_handle)
        if A.dtype.char == 'f':
            self.dotc = _cublas.sdot
            self.nrm2 = _cublas.snrm2
            self.gemm = _cublas.sgemm
        elif A.dtype.char == 'd':
            self.dotc = _cublas.ddot
            self.nrm2 = _cublas.dnrm2
            self.gemm = _cublas.dgemm
        elif A.dtype.char == 'F':
            self.dotc = _cublas.cdotc
            self.nrm2 = _cublas.scnrm2
            self.gemm = _cublas.cgemm
        elif A.dtype.char == 'D':
            self.dotc = _cublas.zdotc
            self.nrm2 = _cublas.dznrm2
            self.gemm = _cublas.zgemm
        else:
            raise TypeError('invalid dtype ({})'.format(A.dtype))
        if csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
            self.cusparse_handle = device.get_cusparse_handle()
            self.spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
            self.spmv_alpha = numpy.array(1.0, A.dtype)
            self.spmv_beta = numpy.array(0.0, A.dtype)
            self.spmv_cuda_dtype = cusparse._dtype_to_DataType(A.dtype)
            self.spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
        else:
            self.cusparse_handle = None
        self.v = cupy.empty((self.n,), dtype=A.dtype)
        self.u = cupy.empty((self.n,), dtype=A.dtype)
        self.uu = cupy.empty((self.ncv,), dtype=A.dtype)

    def update(self, i_start, i_end):
        assert 0 <= i_start and i_end <= self.ncv
        if self.update_impl == 'fast':
            return self._update_fast(i_start, i_end)
        else:
            return self._update_asis(i_start, i_end)

    def _update_asis(self, i_start, i_end):
        for i in range(i_start, i_end):
            u = self.A @ self.V[i]
            cublas.dotc(self.V[i], u, out=self.alpha[i])
            u -= u.T @ self.V[:i+1].conj().T @ self.V[:i+1]
            cublas.nrm2(u, out=self.beta[i])
            if i >= i_end - 1:
                break
            self.V[i+1] = u / self.beta[i]
        return u

    def _update_fast(self, i_start, i_end):
        self._spmv_init()
        self.v[...] = self.V[i_start]
        for i in range(i_start, i_end):
            self._spmv(i)
            self._dotc(i)
            self._orthogonalize(i)
            self._norm(i)
            if i >= i_end - 1:
                break
            self._normalize(i)
        self._spmv_fin()
        return self.u.copy()

    def _spmv(self, i):
        if self.cusparse_handle is None:
            self.u[...] = self.A @ self.v
        else:
            _cusparse.spMV(
                self.cusparse_handle, self.spmv_op_a,
                self.spmv_alpha.ctypes.data, self.spmv_desc_A.desc,
                self.spmv_desc_v.desc, self.spmv_beta.ctypes.data,
                self.spmv_desc_u.desc, self.spmv_cuda_dtype,
                self.spmv_alg, self.spmv_buff.data.ptr)

    def _spmv_init(self):
        if self.cusparse_handle is None:
            return
        self.spmv_desc_A = cusparse.SpMatDescriptor.create(self.A)
        self.spmv_desc_v = cusparse.DnVecDescriptor.create(self.v)
        self.spmv_desc_u = cusparse.DnVecDescriptor.create(self.u)
        buff_size = _cusparse.spMV_bufferSize(
            self.cusparse_handle, self.spmv_op_a,
            self.spmv_alpha.ctypes.data, self.spmv_desc_A.desc,
            self.spmv_desc_v.desc, self.spmv_beta.ctypes.data,
            self.spmv_desc_u.desc, self.spmv_cuda_dtype, self.spmv_alg)
        self.spmv_buff = cupy.empty(buff_size, cupy.int8)

    def _spmv_fin(self):
        if self.cusparse_handle is None:
            return
        # Note: I would like to reuse descriptors and working buffer on the
        # next update, but I gave it up because it sometimes caused illegal
        # memory access error.
        del self.spmv_desc_A
        del self.spmv_desc_v
        del self.spmv_desc_u
        del self.spmv_buff

    def _dotc(self, i):
        _cublas.setPointerMode(self.cublas_handle,
                               _cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            self.dotc(self.cublas_handle, self.n, self.v.data.ptr, 1,
                      self.u.data.ptr, 1,
                      self.alpha.data.ptr + i * self.alpha.itemsize)
        finally:
            _cublas.setPointerMode(self.cublas_handle,
                                   self.cublas_pointer_mode)

    def _orthogonalize(self, i):
        self.gemm(self.cublas_handle,
                  _cublas.CUBLAS_OP_C, _cublas.CUBLAS_OP_N,
                  1, i+1, self.n,
                  1.0, self.u.data.ptr, self.n, self.V.data.ptr, self.n,
                  0.0, self.uu.data.ptr, 1)
        self.gemm(self.cublas_handle,
                  _cublas.CUBLAS_OP_N, _cublas.CUBLAS_OP_C,
                  self.n, 1, i+1,
                  -1.0, self.V.data.ptr, self.n, self.uu.data.ptr, 1,
                  1.0, self.u.data.ptr, self.n)

    def _norm(self, i):
        _cublas.setPointerMode(self.cublas_handle,
                               _cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            self.nrm2(self.cublas_handle, self.n, self.u.data.ptr, 1,
                      self.beta.data.ptr + i * self.beta.itemsize)
        finally:
            _cublas.setPointerMode(self.cublas_handle,
                                   self.cublas_pointer_mode)

    def _normalize(self, i):
        _kernel_normalize(self.u, self.beta, i, self.n, self.v, self.V)


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
        a (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): A real or complex
            array with dimension ``(m, n)``
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

    aH = a.conj().T
    if m >= n:
        aa = aH @ a
    else:
        aa = a @ aH

    if return_singular_vectors:
        w, x = eigsh(aa, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol,
                     return_eigenvectors=True)
    else:
        w = eigsh(aa, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol,
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
        v = aH @ u / s[:n_large]
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
