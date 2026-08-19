from __future__ import annotations

import numpy
import cupy

from cupy import cublas
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface


def eigsh(a, k=6, *, which='LM', v0=None, ncv=None, maxiter=None,
          tol=0, return_eigenvectors=True):
    """
    Find ``k`` eigenvalues and eigenvectors of the real symmetric square
    matrix or complex Hermitian matrix ``A``.

    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues
    with corresponding eigenvectors ``x``.

    Args:
        a (ndarray, spmatrix or LinearOperator): A symmetric square matrix with
            dimension ``(n, n)``. ``a`` must :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        k (int): The number of eigenvalues and eigenvectors to compute. Must be
            ``1 <= k < n``.
        which (str): 'LM' or 'LA' or 'SA'.
            'LM': finds ``k`` largest (in magnitude) eigenvalues.
            'LA': finds ``k`` largest (algebraic) eigenvalues.
            'SA': finds ``k`` smallest (algebraic) eigenvalues.

        v0 (ndarray): Starting vector for iteration. If ``None``, a random
            unit vector is used.
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
    if which not in ('LM', 'LA', 'SA'):
        raise ValueError('which must be \'LM\',\'LA\'or\'SA\' (actual: {})'
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
    if v0 is None:
        u = cupy.random.random((n,)).astype(a.dtype)
        V[0] = u / cublas.nrm2(u)
    else:
        u = v0
        V[0] = v0 / cublas.nrm2(v0)

    # Choose Lanczos implementation, unconditionally use 'fast' for now
    upadte_impl = 'fast'
    if upadte_impl == 'fast':
        lanczos = _lanczos_fast(a, n, ncv)
    else:
        lanczos = _lanczos_asis

    # Lucky-breakdown detection threshold (see _lanczos_checked): decoupling
    # the tridiagonal at beta[i] perturbs the spectrum by at most beta[i], so
    # an O(eps)*||A|| threshold is harmless by construction. (A sqrt(eps)
    # threshold can exceed legitimate small couplings in float32 and corrupt
    # the trailing Ritz values.)
    break_rtol = 64.0 * float(numpy.finfo(a.dtype).eps)
    # Restart-reseed bias (see _restart_ortho): pull reseeds out of the
    # operator's null space -- except for 'SA', where the smallest
    # (possibly zero) eigenvalues are the ones being sought.
    bias_op = a if which != 'SA' else None
    la_shift = (which == 'LA')

    # Lanczos iteration
    anorm, bias_shift = _lanczos_checked(a, lanczos, V, u, alpha, beta, 0,
                                         ncv, break_rtol, bias_op, la_shift)

    iter = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s

    # Compute residual
    beta_k = beta[-1] * s[-1, :]
    res = cublas.nrm2(beta_k)

    uu = cupy.empty((k,), dtype=a.dtype)

    while res > tol and iter < maxiter:
        # Setup for thick-restart
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T

        # u -= u.T @ V[:k].conj().T @ V[:k]
        cublas.gemv(_cublas.CUBLAS_OP_C, 1, V[:k].T, u, 0, uu)
        cublas.gemv(_cublas.CUBLAS_OP_N, -1, V[:k].T, uu, 1, u)
        # The deflation above can eat nearly all of u (residual almost inside
        # the locked Ritz span, common in single precision): dividing by the
        # collapsed norm would inject a huge junk vector and wild Ritz values.
        # Reseed with a fresh direction orthogonal to V[:k] instead. (One
        # host float per restart cycle; solve_ritz syncs here anyway.)
        nrm_u = float(cublas.nrm2(u))
        if nrm_u > break_rtol * anorm:
            V[k] = u / nrm_u
        else:
            V[k] = _restart_ortho(V, k, n, V.dtype, bias_op, bias_shift)

        u[...] = a @ V[k]
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        # NaN-proof normalization: beta[k] = ||u||, so the division is finite
        # for beta[k] > 0; an exact zero (restart subspace invariant) yields a
        # zero column, then _lanczos_checked's boundary check (which covers
        # index k) decouples and reseeds it.
        V[k+1] = cupy.where(beta[k] > 0, u / beta[k], u * 0)

        # Lanczos iteration
        anorm, bias_shift = _lanczos_checked(a, lanczos, V, u, alpha, beta,
                                             k + 1, ncv, break_rtol, bias_op,
                                             la_shift)

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
    from cupy_backends.cuda.libs import cusparse as _cusparse
    from cupyx import cusparse

    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
    if A.dtype.char == 'f':
        dotc = _cublas.sdot
        nrm2 = _cublas.snrm2
        gemv = _cublas.sgemv
        axpy = _cublas.saxpy
    elif A.dtype.char == 'd':
        dotc = _cublas.ddot
        nrm2 = _cublas.dnrm2
        gemv = _cublas.dgemv
        axpy = _cublas.daxpy
    elif A.dtype.char == 'F':
        dotc = _cublas.cdotc
        nrm2 = _cublas.scnrm2
        gemv = _cublas.cgemv
        axpy = _cublas.caxpy
    elif A.dtype.char == 'D':
        dotc = _cublas.zdotc
        nrm2 = _cublas.dznrm2
        gemv = _cublas.zgemv
        axpy = _cublas.zaxpy
    else:
        raise TypeError('invalid dtype ({})'.format(A.dtype))

    cusparse_handle = None
    if _csr._is_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT

    v = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    vv = cupy.empty((n,), dtype=A.dtype)
    b = cupy.empty((), dtype=A.dtype)
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

            # Call dotc: alpha[i] = v.conj().T @ u
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1,
                     alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)

            # Orthogonalize: u = u - alpha[i] * v - beta[i - 1] * V[i - 1]
            vv.fill(0)
            b[...] = beta[i - 1]    # cast from real to complex
            _cublas.setPointerMode(
                cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                axpy(cublas_handle, n,
                     alpha.data.ptr + i * alpha.itemsize,
                     v.data.ptr, 1, vv.data.ptr, 1)
                axpy(cublas_handle, n,
                     b.data.ptr,
                     V[i - 1].data.ptr, 1, vv.data.ptr, 1)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            axpy(cublas_handle, n,
                 mone.ctypes.data,
                 vv.data.ptr, 1, u.data.ptr, 1)

            # Reorthogonalize: u -= V @ (V.conj().T @ u)
            gemv(cublas_handle, _cublas.CUBLAS_OP_C,
                 n, i + 1,
                 one.ctypes.data, V.data.ptr, n,
                 u.data.ptr, 1,
                 zero.ctypes.data, uu.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N,
                 n, i + 1,
                 mone.ctypes.data, V.data.ptr, n,
                 uu.data.ptr, 1,
                 one.ctypes.data, u.data.ptr, 1)
            alpha[i] += uu[i]

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

            # Normalize. beta[i] = ||u||, so u/beta[i] stays finite for any
            # beta[i] > 0; the kernel emits 0 instead of dividing when
            # beta[i] == 0 (exact lucky breakdown), so the sweep completes
            # NaN-free and _lanczos_checked repairs it at the sweep boundary.
            _kernel_normalize(u, beta, i, n, v, V)

    return aux


_kernel_normalize = cupy.ElementwiseKernel(
    'T u, raw S beta, int32 j, int32 n', 'T v, raw T V',
    # beta[j] = ||u||, so u/beta[j] is finite whenever beta[j] > 0; on an
    # EXACT lucky breakdown (beta[j] == 0) emit 0 instead of 0/0 = NaN --
    # the zero column is detected and repaired at the sweep boundary
    # (_lanczos_checked), keeping this inner loop free of host syncs.
    'S b = beta[j]; v = (b > (S)0) ? (u / b) : (u * (S)0);'
    ' V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)


def _lanczos_checked(a, lanczos, V, u, alpha, beta, i_start, i_end,
                     break_rtol, bias_op=None, la_shift=False):
    """Run a Lanczos sweep; detect and repair lucky breakdowns at the sweep
    boundary.

    beta[i] ~ 0 means A @ v landed in span(V), i.e. an invariant subspace was
    found; normalizing by it yields NaNs / wrong results on degenerate or
    rank-deficient spectra (gh-6446, gh-7495, gh-8009). The inner loop stays
    completely free of host syncs (its device-pointer-mode design): an exact
    breakdown produces a zero column (see _kernel_normalize) and a near
    breakdown a finite junk-but-orthonormalized column, both harmless within
    the sweep. HERE, on the host copy of (alpha, beta) that the Ritz solve
    needs anyway, positions with beta[i] < break_rtol * ||A||_est are
    decoupled (beta[i] = 0 -- a spectrum perturbation of at most beta[i],
    machine-level by the threshold choice) and V[i+1] is reseeded with a
    deterministic fresh vector orthogonal to V[:i+1]; the sweep re-runs from
    there so the iteration keeps exploring. Healthy problems detect nothing
    and pay zero overhead; a fully degenerate spectrum costs at most
    O(i_end - i_start) partial re-sweeps.

    The check range starts at i_start - 1, so it also covers the thick
    restart's own normalization V[k+1] = u / beta[k] (repaired the same way).
    Returns the ||A|| estimate for the driver's own guard on V[k].
    """
    n = V.shape[1]
    start = i_start
    repaired = set()
    while True:
        lanczos(a, V, u, alpha, beta, start, i_end)
        alpha_np = cupy.asnumpy(alpha)
        alpha_h = numpy.abs(alpha_np)
        beta_h = numpy.abs(cupy.asnumpy(beta))
        # Gershgorin-style row-sum estimate of ||T|| ~ ||A||, built by
        # walking the rows IN ORDER and stopping at the first breakdown:
        # rows past an exhausted subspace hold roundoff junk whose magnitude
        # can dwarf ||A|| -- folding them into the estimate would inflate
        # the threshold and misfire on legitimate large couplings.
        # Vectorized: the running max IS a cumulative maximum, so the whole
        # ordered walk is one numpy pass -- no per-ncv Python loop on the
        # healthy path (which produces no candidates at all).
        m = i_end - 1
        # float64 accumulation: alpha/beta past an exhausted subspace can be
        # huge in float32 (the junk this walk exists to exclude) and the
        # row sums would overflow to inf before we ever get to ignore them.
        rows = alpha_h[:m].astype(numpy.float64) + beta_h[:m]
        rows[1:] += beta_h[:m - 1]
        anorm_run = numpy.maximum.accumulate(rows)
        # Non-strict (<=) so an all-zero block (beta and anorm both 0, e.g.
        # v0 in the null space of a singular A) is still caught and
        # reseeded, not silently left as zero Ritz values.
        # LA reseed shift: bias by (A + anorm*I) ONLY when zero may itself
        # be an LA target, i.e. no positive Rayleigh quotient has been seen
        # (v^H A v <= 0 for every Lanczos vector so far, so A looks negative
        # semidefinite and its null space sits at the TOP of the spectrum).
        # On a positive-semidefinite or indefinite operator the shift would
        # instead let null-space candidates pass _accept and waste Krylov
        # slots on eigenvalue-0 directions that are not targets, while the
        # plain bias correctly annihilates them.
        hits = numpy.flatnonzero(beta_h[:m] <= break_rtol * anorm_run)
        lo = max(i_start - 1, 0)
        p = None
        for i in hits:                       # typically empty
            i = int(i)
            if i >= lo and i not in repaired:
                p = i
                break
        # ||A|| estimate and the shift decision must use only the PREFIX up
        # to the breakdown -- rows after it hold roundoff junk (this is the
        # same reason the walk is ordered). Equivalent to the scalar loop,
        # which broke out at p and never looked further.
        end = m - 1 if p is None else p
        anorm = float(anorm_run[end])
        shift = (anorm if (la_shift
                           and float(alpha_np[:end + 1].real.max()) <= 0.0)
                 else 0.0)
        if p is None:
            return anorm, shift
        beta[p] = 0
        V[p + 1] = _restart_ortho(V, p + 1, n, V.dtype, bias_op, shift)
        repaired.add(p)
        start = p + 1


def _restart_ortho(V, m, n, dtype, bias_op=None, bias_shift=0.0):
    # Deterministic fresh unit vector orthogonal to V[:m], used to restart
    # the Lanczos recurrence after a lucky breakdown. Start from the
    # canonical basis vector least represented in span(V[:m]) -- the rows of
    # V[:m] are orthonormal, so the residual ||(I - P) e_j||^2 is
    # 1 - ||V[:m, j]||^2 and its maximum over j is >= 1 - m/n > 0 -- then
    # orthogonalize with two classical Gram-Schmidt passes. No RNG: the
    # result depends only on V, keeping eigsh/svds output deterministic.
    #
    # bias_op: when the operator has a large null space (e.g. svds on a
    # low-rank matrix works through A^H A whose null fraction is
    # 1 - rank/n), a canonical basis vector is almost entirely null and the
    # restart wastes Krylov slots exploring eigenvalue-0 directions
    # (gh-8009's reproducer: 100x1000 rank 5). One application of the
    # operator kills all null components, so seed with bias_op @ e_j
    # instead (still deterministic); fall back to the unbiased e_j when the
    # result lies in span(V[:m]) -- then the nonzero spectrum is exhausted
    # and the null space is exactly what remains to explore.
    # Robustness of the acceptance test: with orthonormal rows the argmin
    # candidate's remainder is >= 1/sqrt(n) (sum of column norms^2 is
    # m <= n-1), but transient repair states can hold zero or junk rows,
    # for which CGS is not a projector and no lower bound exists -- a
    # remainder can cancel arbitrarily small, and normalizing it would
    # amplify CGS roundoff into a poorly-orthogonal noise direction (or
    # 0/0 = NaN on exact cancellation). So: accept a remainder only above
    # a dtype-aware floor (sqrt(eps), far above the CGS noise floor in
    # both precisions), try a few canonical vectors in ascending
    # column-norm order, and if all fail return the plain canonical
    # vector: unit, finite and deterministic. Its lost orthogonality is
    # absorbed by the sweep's own per-iteration reorthogonalization and
    # the next boundary check -- always preferable to NaN or noise.
    # Candidate ladder (all deterministic, first acceptable wins):
    #   1. biased DENSE probe, varied with m -- a canonical e_j fails on
    #      operators whose null space is coordinate-aligned (bias_op @ e_j
    #      is then exactly 0: gh-8009's literal reproducer makes A^H A a
    #      coordinate projector), and a FIXED dense probe is spanned after
    #      its first acceptance, so the probe must change per reseed;
    #   2. biased canonical walk over the least-represented columns;
    #   3. unbiased canonical walk (correct once the operator's range is
    #      exhausted: the null space is exactly what remains to explore);
    #   4. nothing acceptable -> raise (fail closed) rather than store a
    #      vector that is not numerically orthogonal to V[:m].
    # bias_shift: biasing by (bias_op + shift*I) instead of bias_op keeps
    # zero eigenvalues alive when they are legitimate targets -- 'LA' on a
    # negative-semidefinite operator has the null space at the TOP of the
    # spectrum, so the driver passes shift ~ +||A|| there (null maps to
    # shift != 0 while the anti-target bottom maps to ~0). 'LM' passes 0
    # (plain bias_op preserves the |lambda| ordering exactly).
    Vm = V[:m]
    colnorm = cupy.sum(cupy.abs(Vm) ** 2, axis=0)
    order = [int(j) for j in cupy.asnumpy(cupy.argsort(colnorm)[:4])]
    floor = float(numpy.sqrt(numpy.finfo(
        numpy.dtype(dtype).char.lower()).eps))

    def _accept(cand):
        nrm = float(cupy.linalg.norm(cand))
        if nrm == 0.0:
            return None
        w = cand / nrm
        for _ in range(2):
            w = w - Vm.T @ (Vm.conj() @ w)
        nrm = float(cupy.linalg.norm(w))
        return (w / nrm) if nrm > floor else None

    if bias_op is not None:
        d = cupy.cos((m + 1.0) * 0.7 * cupy.arange(1, n + 1)).astype(dtype)
        d /= cupy.linalg.norm(d)
        w = _accept(bias_op @ d + bias_shift * d)
        if w is not None:
            return w
    e = cupy.zeros((n,), dtype=dtype)
    if bias_op is not None:
        for j in order:
            e[...] = 0
            e[j] = 1
            w = _accept(bias_op @ e + bias_shift * e)
            if w is not None:
                return w
    for j in order:
        e[...] = 0
        e[j] = 1
        w = _accept(e)
        if w is not None:
            return w
    # Fail closed. Reaching here means no candidate produced a remainder
    # above the sqrt(eps) floor, i.e. we cannot supply a vector that is
    # numerically orthogonal to V[:m]. Returning a raw (unprojected) e_j
    # would satisfy "unit, finite and deterministic" but not "safe to store
    # in V": the caller would continue under an orthonormality invariant
    # that no longer holds, producing silently wrong Ritz values. With
    # orthonormal rows this is unreachable (the argmin-column remainder is
    # >= sqrt(1 - m/n)), so it only fires from an already-corrupted basis.
    raise RuntimeError(
        'eigsh: Lanczos restart failed -- no direction orthogonal to the '
        'current basis exceeds the {:.1e} floor after {} candidates. The '
        'basis is numerically rank-deficient; try a smaller ncv or a '
        'different v0.'.format(floor, 2 * len(order) + 1))


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
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'LM':
        idx = numpy.argsort(numpy.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]

    elif which == 'SA':
        idx = numpy.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]
    # elif which == 'SM':  #dysfunctional
    #   idx = cupy.argsort(abs(w))
    #   wk = w[idx[:k]]
    #   sk = s[:,idx[:k]]
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
    n_large = above_cutoff.sum().item()
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
