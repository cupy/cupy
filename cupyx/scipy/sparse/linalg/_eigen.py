from __future__ import annotations

import numpy
import cupy

from cupy import cublas
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface

# Seed of the default Lanczos start vector. Drawing it from a fixed seed
# makes eigsh and svds deterministic for a given input, as ARPACK's default
# start is, and independent of the global cupy.random state; the trajectory
# can still be chosen explicitly through v0.
_DEFAULT_V0_SEED = 0


def _default_v0(n, dtype, rs=None):
    """Pseudo-random start vector of length ``n`` drawn from a fixed seed.

    A private ``RandomState`` is used so that neither ``cupy.random.seed``
    nor any other consumer of the global generator changes the result.
    """
    if rs is None:
        rs = cupy.random.RandomState(_DEFAULT_V0_SEED)
    return rs.random_sample((n,)).astype(dtype)


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

        v0 (ndarray): Starting vector for iteration. If ``None``, a
            pseudo-random unit vector drawn from a fixed seed is used, so
            repeated calls on the same input follow the same trajectory
            regardless of the global :mod:`cupy.random` state (as
            :func:`scipy.sparse.linalg.svds` does for its default start).
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

    Raises:
        RuntimeError: If the Lanczos basis loses numerical orthogonality
            beyond repair, so that no correct answer can be produced. This
            replaces silently wrong output (NaN or overflowed Ritz values)
            on such inputs, and is raised from three places: the operator
            norm estimate exceeding what an orthonormal basis can give, a
            restart probe that grows when projected out of the basis, and
            exhaustion of the restart candidate ladder. Degenerate or
            rank-deficient spectra in single precision are the usual
            cause; a smaller ``ncv``, a different ``v0``, or ``float64``
            input is the usual remedy. Note that
            :func:`scipy.sparse.linalg.eigsh` signals its own failure mode
            differently, with ``ArpackNoConvergence``.

    .. seealso:: :func:`scipy.sparse.linalg.eigsh`

    .. note::
        This function uses the thick-restart Lanczos methods
        (https://sdm.lbl.gov/~kewu/ps/trlan.html).

    .. note::
        Degenerate and tightly clustered spectra exhaust the Krylov space
        early (a lucky breakdown). Those breakdowns are detected and
        repaired, which keeps the result correct but costs extra sweeps:
        healthy problems pay a few percent, while a strongly degenerate
        spectrum (e.g. one with only two distinct eigenvalues) can take
        several times longer than the nominal iteration count suggests.

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
        u = _default_v0(n, a.dtype)
        V[0] = u / cublas.nrm2(u)
    else:
        u = v0.copy()          # the driver writes into u; do not mutate v0
        V[0] = u / cublas.nrm2(u)

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
    # Orthogonality tolerance, shared by the sweep-boundary check and the
    # locked-block check: far above a healthy basis (~eps*ncv) and far
    # below the failures it catches (O(1)).
    #
    # Scale note: this margin and the two above (64*eps, and the 8x
    # headroom on the norm bound) are n-INDEPENDENT, while the roundoff
    # they discriminate against grows with n -- worst case ~eps*sqrt(n)
    # for a length-n inner product. In float64 the crossover is around
    # n ~ 1e15, unreachable. In float32 the worst-case model crosses
    # sqrt(eps) near n ~ 1e7, though blocked GPU reductions keep the real
    # error far below that bound (degenerate and late-norm-discovery
    # cases were checked at n = 3e7 float32 during review). Every one of
    # these margins fails in a safe direction: too loose degrades to the
    # pre-guard behaviour, which the orthogonality check still backstops,
    # and too tight raises the documented RuntimeError -- neither returns
    # silent garbage. If float32 at n >> 1e7 ever needs support, the
    # explicit form is max(sqrt(eps), c*eps*sqrt(n)).
    ortho_rtol = float(numpy.sqrt(numpy.finfo(a.dtype).eps))
    # True upper bound on ||A|| for the divergence guard (None for a
    # LinearOperator, whose entries are not available).
    norm_bound = _norm_upper_bound(a)
    # Restart-reseed bias (see _restart_ortho): pull reseeds out of the
    # operator's null space -- except for 'SA', where the smallest
    # (possibly zero) eigenvalues are the ones being sought.
    bias_op = a if which != 'SA' else None
    la_shift = (which == 'LA')

    # Lanczos iteration
    anorm, bias_shift, work = _lanczos_checked(a, lanczos, V, u, alpha, beta,
                                               0, ncv, break_rtol, bias_op,
                                               la_shift,
                                               ortho_rtol=ortho_rtol,
                                               norm_bound=norm_bound)

    iter = work
    w, s, w_host = _eigsh_solve_ritz(alpha, beta, None, k, which)
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
        _repair_locked(a, V, alpha, k, n, ortho_rtol, bias_op,
                       bias_shift, w_host=w_host)

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
            # V[k] is now a fresh direction, not the old residual, so the
            # coupling row beta_k no longer describes it (it is used just
            # below and as the Ritz arrowhead). This branch fires only when
            # the residual collapsed into span(V[:k]), i.e. that span is
            # already numerically invariant, so the true coupling to any
            # direction outside it is at threshold scale. Zero it: T is then
            # exactly block-decoupled -- the same repair _lanczos_checked
            # makes with beta[p] = 0.
            beta_k[...] = 0

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
        anorm, bias_shift, work = _lanczos_checked(a, lanczos, V, u, alpha,
                                                   beta, k + 1, ncv,
                                                   break_rtol, bias_op,
                                                   la_shift,
                                                   ortho_rtol=ortho_rtol,
                                                   norm_bound=norm_bound)

        # +1 for the u = a @ V[k] above; equals the former ncv - k when no
        # repair re-sweep ran.
        iter += work + 1
        w, s, w_host = _eigsh_solve_ritz(alpha, beta, beta_k, k,
                                         which)
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
    # j and n are int64 so the strided write offset (j+1) * n is computed
    # in 64-bit; an int32 product wraps for n * ncv >= 2**31 (e.g. dim
    # 2**28 with the default ncv) and corrupts memory.
    'T u, raw S beta, int64 j, int64 n', 'T v, raw T V',
    # beta[j] = ||u||, so u/beta[j] is finite whenever beta[j] > 0; on an
    # EXACT lucky breakdown (beta[j] == 0) emit 0 instead of 0/0 = NaN --
    # the zero column is detected and repaired at the sweep boundary
    # (_lanczos_checked), keeping this inner loop free of host syncs.
    'S b = beta[j]; v = (b > (S)0) ? (u / b) : (u * (S)0);'
    ' V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize'
)


# The row-sum estimate of ||T|| is bounded by 3 * ||T|| (a tridiagonal row
# has at most three entries) and ||T|| <= ||A|| for orthonormal V, so a
# healthy estimate can never exceed 3 * ||A||. Compared against a true
# upper bound on ||A||, anything past this factor is therefore PROVABLE
# corruption rather than a tuned threshold -- which matters because a
# legitimate late discovery of ||A|| can be arbitrarily large: an operator
# with wide dynamic range and a v0 orthogonal to its dominant eigenspace
# (what deflation workflows produce) makes the first sweep honestly miss
# ||A|| and a later reseed find it.
_DIVERGE_RTOL = 8.0


def _norm_upper_bound(a):
    """Max absolute row sum of ``a``, an upper bound on ``||a||_2``.

    Returns None when the operator's entries are not available (e.g. a
    LinearOperator), in which case no absolute bound can be formed and the
    caller falls back to checking only for non-finite estimates.
    """
    try:
        if isinstance(a, cupy.ndarray):
            return float(cupy.abs(a).sum(axis=1).max())
        if _csr._is_csr(a) or getattr(a, 'format', None) in (
                'csr', 'csc', 'coo'):
            # |A| @ 1 gives the absolute row sums; A is Hermitian here, so
            # a CSC/COO layout yields the same bound.
            b = a.tocsr()
            ones = cupy.ones(b.shape[1], dtype=b.dtype)
            return float(cupy.abs(cupy.abs(b) @ ones).max())
    except Exception:      # pragma: no cover - bound is optional
        return None
    return None


def _lanczos_checked(a, lanczos, V, u, alpha, beta, i_start, i_end,
                     break_rtol, bias_op=None, la_shift=False,
                     ortho_rtol=None, norm_bound=None):
    """Run a Lanczos sweep; detect and repair lucky breakdowns at the sweep
    boundary.

    beta[i] ~ 0 means A @ v landed in span(V) -- an invariant subspace -- and
    normalizing by it yields NaNs or wrong results on degenerate and
    rank-deficient spectra (gh-6446, gh-7495, gh-8009). The inner loop stays
    free of host syncs: an exact breakdown emits a zero column
    (_kernel_normalize), a near one a finite column, both harmless within the
    sweep. Here, on the host copy of (alpha, beta) the Ritz solve needs
    anyway, positions with beta[i] < break_rtol * ||A||_est are decoupled
    (beta[i] = 0, perturbing the spectrum by at most beta[i]) and V[i+1] is
    reseeded orthogonal to V[:i+1]; the sweep re-runs from there. The range
    starts at i_start - 1, so it also covers the thick restart's own
    V[k+1] = u / beta[k].

    Cost: detection is a few host ops per sweep (2-6% end to end on healthy
    problems, measured); a degenerate spectrum can trigger up to
    O(i_end - i_start) partial re-sweeps, which is why the matvec count is
    returned rather than assumed.

    Returns (||A|| estimate for the driver's guard on V[k], LA reseed shift,
    matvecs performed including re-sweeps).
    """
    n = V.shape[1]
    if ortho_rtol is None:
        ortho_rtol = float(numpy.sqrt(numpy.finfo(
            numpy.dtype(V.dtype).char.lower()).eps))
    start = i_start
    repaired = set()
    # Matvecs actually performed, including repair re-sweeps, so the driver's
    # maxiter still bounds work on degenerate spectra (where a sweep can be
    # re-run up to O(i_end - i_start) times).
    work = 0
    while True:
        work += i_end - start
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
        if m == 0:
            # ncv == 1, reachable only at n == 2: no beta[i] has both
            # neighbours in range, so there is nothing to check or repair
            # (position 0 is covered by the driver's own nrm_u guard).
            # Estimate ||A|| from the single row; the cumulative walk below
            # would index an empty array.
            anorm = float(alpha_h[0] + beta_h[0])
            shift = (anorm if (la_shift and float(alpha_np[0].real) <= 0.0)
                     else 0.0)
            return anorm, shift, work
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
        # The absolute test above catches a breakdown only while the
        # residual is still AT the roundoff floor. On a spectrum whose
        # Krylov dimension is far below ncv (e.g. two distinct
        # eigenvalues) the floor itself grows as roundoff is amplified --
        # measured 5e-14 -> 5e-8 over twenty steps -- so after the first
        # couple of repairs no fixed multiple of eps*||A|| fires again,
        # and the basis quietly fills with amplified noise. What does not
        # drift is the invariant the method actually rests on: every
        # stored row must be numerically orthogonal to the rows before
        # it. Verify it directly with one Gram matrix (a single BLAS3
        # call on the ncv x n basis), gated on a cheap host-side trigger
        # so a healthy sweep -- which never produces a coupling far below
        # the operator scale -- pays nothing.
        # Trigger on the ACTIVE range only: the thick restart deliberately
        # sets beta[:k] = 0, so scanning from 0 would fire this check on
        # every restart of a perfectly healthy problem (measured: up to
        # +23% on an FP64-limited card, where the Gram is a float64 GEMM).
        if bool((beta_h[lo:m] <= ortho_rtol * anorm_run[lo:m]).any()):
            Vm = V[:i_end]
            gram = cupy.tril(cupy.abs(Vm @ Vm.conj().T), -1)
            worst = cupy.asnumpy(gram.max(axis=1))
            # Row j lost orthogonality => V[j] was produced by normalizing
            # a collapsed residual at step j-1: that is the breakdown.
            for j in numpy.flatnonzero(worst > ortho_rtol):
                q = int(j) - 1
                if q >= lo and q not in repaired and (p is None or q < p):
                    p = q
                    break
        # ||A|| estimate and the shift decision must use only the PREFIX up
        # to the breakdown -- rows after it hold roundoff junk (this is the
        # same reason the walk is ordered). Equivalent to the scalar loop,
        # which broke out at p and never looked further.
        end = m - 1 if p is None else p
        anorm = float(anorm_run[end])
        # Fail closed on an estimate that cannot come from an orthonormal
        # basis: with ||T|| <= ||A|| and the row-sum estimate <= 3 ||T||,
        # exceeding 8x a true upper bound on ||A|| is impossible without
        # corruption. A non-finite estimate is corrupt by inspection. When
        # no bound is available (LinearOperator) only the latter applies --
        # a relative test against the first sweep would misfire on the
        # legitimate late-discovery case described at _DIVERGE_RTOL.
        if not numpy.isfinite(anorm):
            raise RuntimeError(
                'eigsh: Lanczos recurrence diverged (the ||A|| estimate is '
                'not finite). The Krylov basis is numerically '
                'rank-deficient; try a smaller ncv, a different v0, or '
                'float64.')
        if norm_bound is not None and anorm > _DIVERGE_RTOL * norm_bound:
            raise RuntimeError(
                'eigsh: Lanczos recurrence diverged (||A|| estimate {:.3e} '
                'exceeds {:.0f}x the operator norm bound {:.3e}, which an '
                'orthonormal basis cannot produce). The Krylov basis is '
                'numerically rank-deficient; try a smaller ncv, a '
                'different v0, or float64.'.format(
                    anorm, _DIVERGE_RTOL, norm_bound))
        shift = (anorm if (la_shift
                           and float(alpha_np[:end + 1].real.max()) <= 0.0)
                 else 0.0)
        if p is None:
            return anorm, shift, work
        beta[p] = 0
        V[p + 1] = _restart_ortho(V, p + 1, n, V.dtype, bias_op, shift)
        repaired.add(p)
        start = p + 1


def _repair_locked(a, V, alpha, k, n, ortho_rtol, bias_op, bias_shift,
                   w_host=None):
    # The locked block V[:k] = (V.T @ s).T inherits orthonormality from V
    # and from the Ritz basis s -- but only when BOTH are sound. On an
    # eigenvalue of multiplicity > 1 the Ritz basis is degenerate by
    # construction and several columns of s can name the same physical
    # direction, so V[:k] can hold duplicate rows (measured: worst
    # off-diagonal 4.0e-01 immediately after locking, where orthonormality
    # demands ~1e-16). The sweep-boundary check cannot see this, because it
    # only repairs positions at or after i_start - 1; a duplicate locked
    # here would survive to poison every later reseed. So verify the block
    # where it is created: one Gram of (k+1) x n per restart, far cheaper
    # than the ncv x n check per sweep.
    #
    # A duplicated Ritz vector carries no information the block does not
    # already have, so replacing it with a fresh orthogonal direction loses
    # nothing. beta[:k] is zero here, i.e. T is block-decoupled, so each
    # locked row is its own 1x1 block and alpha[j] must be updated to the
    # new direction's Rayleigh quotient to keep T consistent.
    #
    # Gated on the Ritz values, which the driver already holds on the host:
    # duplicate Ritz VECTORS can only arise from a degenerate Ritz basis,
    # i.e. from (near-)repeated Ritz VALUES. That test is free -- k host
    # floats -- whereas running the Gram unconditionally costs a device
    # sync per restart, measured at +9-13% end to end on healthy problems.
    if k < 2:
        return
    if w_host is not None:
        wr = numpy.asarray(w_host, dtype=numpy.float64).real
        scale = float(numpy.abs(wr).max()) if wr.size else 0.0
        # Sort the SIGNED values: a +/-lambda pair is not degenerate, and
        # sorting magnitudes would make an indefinite spectrum (common
        # under 'LM') fire this gate for nothing.
        if scale == 0.0 or not numpy.any(
                numpy.diff(numpy.sort(wr)) <= ortho_rtol * scale):
            return
    # Only the locked rows: the driver overwrites V[k] a few lines below,
    # and its overlap with the locked block is |s[k, j]|, normally far
    # above the tolerance -- including it would reseed a row that is about
    # to be discarded, at the cost of a matvec and a sync.
    Vk = V[:k]
    gram = cupy.tril(cupy.abs(Vk @ Vk.conj().T), -1)
    worst = cupy.asnumpy(gram.max(axis=1))
    for j in numpy.flatnonzero(worst > ortho_rtol):
        j = int(j)
        V[j] = _restart_ortho(V, j, n, V.dtype, bias_op, bias_shift)
        alpha[j] = cupy.inner(V[j].conj(), a @ V[j])


def _restart_ortho(V, m, n, dtype, bias_op=None, bias_shift=0.0):
    # Deterministic unit vector orthogonal to V[:m], used to restart the
    # recurrence after a lucky breakdown.
    #
    # INVARIANT: what this returns is numerically orthogonal to V[:m], or it
    # raises. Nothing else may be stored in V.
    #
    # No RNG -- the result depends only on V, keeping eigsh/svds output
    # deterministic. A candidate is accepted only if two classical
    # Gram-Schmidt passes leave a remainder above a sqrt(eps) floor: a
    # transient repair state can hold zero or junk rows, for which CGS is not
    # a projector and a remainder can cancel to noise (or 0/0 = NaN).
    #
    # bias_op (the operator; None for 'SA'): one application annihilates
    # null-space components, which a canonical e_j is mostly made of when the
    # operator has a large null space (svds on a low-rank matrix, gh-8009).
    # bias_shift: bias by (bias_op + shift*I) so zero eigenvalues survive when
    # they are legitimate targets -- 'LA' on a negative-semidefinite operator
    # has the null space at the TOP of the spectrum; 'LM' passes 0.
    #
    # Ladder, first acceptable candidate wins:
    #   1. biased dense probe, varied with m -- a canonical e_j fails when the
    #      null space is coordinate-aligned (bias_op @ e_j is then exactly 0),
    #      and a fixed probe is spanned after its first acceptance;
    #   2. biased canonical walk over the least-represented columns;
    #   3. unbiased canonical walk -- correct once the operator's range is
    #      exhausted, since the null space is then what remains to explore;
    #   4. nothing acceptable -> raise (fail closed).
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
        if nrm > 1.0 + floor:
            # Projecting a unit vector out of an orthonormal V[:m] cannot
            # lengthen it. A remainder above 1 therefore proves V is no
            # longer orthonormal, so no reseed drawn from it can restore
            # the invariant -- fail closed instead of amplifying roundoff
            # into the basis (measured remainders of 9-60 immediately
            # before runaway Ritz values).
            raise RuntimeError(
                'eigsh: Lanczos basis lost orthogonality (a unit probe '
                'grew to {:.3e} when projected out of the first {} '
                'rows). The recurrence cannot be restarted safely; try a '
                'smaller ncv, a different v0, or float64.'.format(nrm, m))
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
    # Fail closed: no candidate cleared the floor, so the invariant above
    # cannot be met. Unreachable from orthonormal rows (the argmin-column
    # remainder is >= sqrt(1 - m/n)); it only fires from a corrupted basis.
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
    # wk is already on the host here; handing it back lets the caller
    # test for degeneracy without paying a device sync.
    return cupy.array(wk), cupy.array(sk), wk


def svds(a, k=6, *, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True):
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
        v0 (ndarray): Starting vector for iteration, of length
            ``min(a.shape)`` as in :func:`scipy.sparse.linalg.svds`. If
            ``None``, a pseudo-random vector drawn from a fixed seed is
            used (see :func:`eigsh`).
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
                     tol=tol, v0=v0, return_eigenvectors=True)
    else:
        w = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol,
                  v0=v0, return_eigenvectors=False)

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
    # svds calls this twice (for u and for v); each call restarts from the
    # same seed, so for m == n the pre-projection draws coincide and only
    # the projections onto the two different bases separate them.
    rs = cupy.random.RandomState(_DEFAULT_V0_SEED)
    for i in range(n, n + n_aug):
        v = _default_v0(m, x.dtype, rs)
        v -= v @ y[:, :i].conj() @ y[:, :i].T
        y[:, i] = v / cupy.linalg.norm(v)
    return y
