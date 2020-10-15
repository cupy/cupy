import numpy
import cupy


def eigsh(a, k=6, which='LM', ncv=None, maxiter=None, tol=0,
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
            ``k + 1 < ncv < n``.
        maxiter (int): Maximum number of Lanczos update iterations.
        tol (float): Tolerance for relative residuals ``||Ax - wx|| / ||wx||``.
            If ``0``, default tolerence is used.
        return_eigenvectors (bool): If ``True``, returns eigenvectors in
            addition to eigenvalue.s

    Returns:
        tuple:
            If ``return_eigenvectors is True``, it returns ``w`` and ``x``
            where ``w`` is eigenvalues and ``x`` is eigenvectors. Otherwise,
            it returns only ``w``.

    .. seealso:: :func:`scipy.sparse.linalg.eigsh`

    .. note::
        This function uses the thick-restart Lanczos methos
        (https://sdm.lbl.gov/~kewu/ps/trlan.html).

    """
    m = a.shape[0]
    if ncv is None:
        ncv = min(max(8 * k, 16), m - 1)
    if maxiter is None:
        maxiter = 10 * m
    if tol == 0:
        tol = numpy.finfo(a.dtype).eps * 64

    alpha = numpy.zeros((ncv, ), dtype=a.dtype)
    beta = numpy.zeros((ncv, ), dtype=a.dtype)
    V = cupy.empty((ncv, m), dtype=a.dtype)

    # Set initial vector
    u = cupy.random.random((m, )).astype(a.dtype)
    v = u / cupy.linalg.norm(u)
    V[0] = v

    # Lanczos iteration
    u = _eigsh_lanczos_update(a, V, v, alpha, beta, 0, ncv)
    iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    # Compute relative residual
    r = a @ x - x @ cupy.diag(w)
    res = cupy.linalg.norm(r) / cupy.linalg.norm(w)

    while res > tol:
        # Setup for thick-restart
        beta[:k] = 0
        alpha[:k] = cupy.asnumpy(w)
        V[:k] = x.T

        u = u - u.T @ V[:k].conj().T @ V[:k]
        beta_last = cupy.linalg.norm(u)
        v = u / beta_last
        V[k] = v
        beta_k = beta_last * s[ncv-1, :]

        u = a @ v
        alpha[k] = v.conj().T @ u
        u = u - alpha[k] * v
        u = u - V[:k].T @ beta_k
        u = u - u.T @ V[:k+1].conj().T @ V[:k+1]
        beta[k] = cupy.linalg.norm(u)
        v = u / beta[k]
        V[k+1] = v

        # Lanczos iteration
        u = _eigsh_lanczos_update(a, V, v, alpha, beta, k+1, ncv)
        iter += ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s

        # Compute relative residual
        r = a @ x - x @ cupy.diag(w)
        res = cupy.linalg.norm(r) / cupy.linalg.norm(w)

        if iter >= maxiter:
            break

    idx = cupy.argsort(w)
    if a.dtype.char in 'FD':
        idx = idx[::-1]
    w = w[idx]
    x = x[:, idx]

    return w, x


def _eigsh_lanczos_update(A, V, v, alpha, beta, j_start, ncv):
    for j in range(j_start, ncv):
        u = A @ v
        alpha[j] = v.conj().T @ u
        u = u - alpha[j] * v
        if j > 0:
            u = u - beta[j-1] * V[j-1]
        if j >= ncv - 1:
            break
        u = u - u.T @ V[:j+1].conj().T @ V[:j+1]
        beta[j] = cupy.linalg.norm(u)
        v = u / beta[j]
        V[j+1] = v
    return u


def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    t = cupy.diag(alpha)
    t = t + cupy.diag(beta[:-1], k=1)
    t = t + cupy.diag(beta[:-1], k=-1)
    if beta_k is not None:
        t[k, :k] = beta_k
        t[:k, k] = beta_k
    w, s = cupy.linalg.eigh(t)

    # pick-up k ritz-values and ritz-vectors
    if which == 'LA':
        wk = w[-k:]
        sk = s[:, -k:]
    elif which == 'LM':
        idx = cupy.argsort(cupy.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]

    return wk, sk
