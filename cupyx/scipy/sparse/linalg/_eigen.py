import numpy
import cupy


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
        ncv = min(max(8 * k, 20), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = numpy.finfo(a.dtype).eps

    alpha = cupy.zeros((ncv, ), dtype=a.dtype)
    beta = cupy.zeros((ncv, ), dtype=a.dtype)
    V = cupy.empty((ncv, n), dtype=a.dtype)

    # Set initial vector
    u = cupy.random.random((n, )).astype(a.dtype)
    v = u / cupy.linalg.norm(u)
    V[0] = v

    # Lanczos iteration
    u = _eigsh_lanczos_update(a, V, alpha, beta, 0, ncv)
    iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    # Compute residual
    beta_k = beta[-1] * s[-1, :]
    res = cupy.linalg.norm(beta_k)

    while res > tol and iter < maxiter:
        # Setup for thick-restart
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T

        u -= u.T @ V[:k].conj().T @ V[:k]
        v = u / cupy.linalg.norm(u)
        V[k] = v

        u = a @ v
        alpha[k] = v.conj().T @ u
        u -= alpha[k] * v
        u -= V[:k].T @ beta_k
        u -= u.T @ V[:k+1].conj().T @ V[:k+1]
        beta[k] = cupy.linalg.norm(u)
        v = u / beta[k]
        V[k+1] = v

        # Lanczos iteration
        u = _eigsh_lanczos_update(a, V, alpha, beta, k+1, ncv)
        iter += ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s

        # Compute residual
        beta_k = beta[-1] * s[-1, :]
        res = cupy.linalg.norm(beta_k)

    if return_eigenvectors:
        idx = cupy.argsort(w)
        return w[idx], x[:, idx]
    else:
        return cupy.sort(w)


def _eigsh_lanczos_update(A, V, alpha, beta, i_start, i_end):
    v = V[i_start]
    for i in range(i_start, i_end):
        u = A @ v
        alpha[i] = v.conj().T @ u
        u -= alpha[i] * v
        if i > 0:
            u -= beta[i-1] * V[i-1]
        u -= u.T @ V[:i+1].conj().T @ V[:i+1]
        beta[i] = cupy.linalg.norm(u)
        if i >= i_end - 1:
            break
        v = u / beta[i]
        V[i+1] = v
    return u


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
            If ``return_singular_vectors is True``, it returns ``u``, ``s`` and
            ``vt`` where ``u`` is left singular vectors, ``s`` is singular
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
