from __future__ import annotations

import functools
import unittest
import warnings

import numpy
import pytest
try:
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.stats
except ImportError:
    pass

import cupy
from cupyx import cusparse, cusolver
from cupy import testing
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy.testing import _condition
from cupyx.scipy import sparse
import cupyx.scipy.sparse.linalg  # NOQA


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
@pytest.mark.skipif(runtime.is_hip, reason='lsqr not supported')
class TestLsqr(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(50, 50, density=0.2, data_rvs=rvs)
        self.b = numpy.random.randint(15, size=50)

    def test_size(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.append(self.b, [1]), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    def test_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    @_condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]

    @_condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray(self, xp, sp):
        A = xp.array(self.A.toarray(), dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]


@testing.parameterize(*testing.product({
    'ord': [None, -numpy.inf, -2, -1, 0, 1, 2, 3, numpy.inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'axis': [None, (0, 1), (1, -2)],
}))
@testing.with_requires('scipy')
class TestMatrixNorm:

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,
                                               NotImplementedError))
    def test_matrix_norm(self, xp, sp):
        if runtime.is_hip and self.ord in (1, -1, numpy.inf, -numpy.inf):
            pytest.xfail('csc spmv is buggy')
        if self.ord == 2:
            pytest.xfail('ord=2 is not implemented in cupy')
        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)


@testing.parameterize(*testing.product({
    'ord': [None, -numpy.inf, -2, -1, 0, 1, 2, numpy.inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'transpose': [True, False],
    'axis': [0, (1,), (-2,), -1],
})
)
@testing.with_requires('scipy')
class TestVectorNorm:

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,))
    def test_vector_norm(self, xp, sp):
        if runtime.is_hip:
            if (self.axis in (0, (-2,))
                    and self.ord in (-2, -1, 0, 1, 2, None)):
                pytest.xfail('csc spmv is buggy')

        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        if self.transpose:
            b = b.T
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)

# TODO : TestVsNumpyNorm


@testing.parameterize(*testing.product({
    'which': ['LM', 'LA', 'SA'],
    'k': [3, 6, 12],
    'return_eigenvectors': [True, False],
    'use_linear_operator': [True, False],
}))
@testing.with_requires('scipy')
class TestEigsh:
    n = 30
    density = 0.33
    tol = {numpy.float32: 1e-5, numpy.complex64: 1e-5, 'default': 1e-12}
    res_tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a * a.conj().T
        return a

    def _test_eigsh(self, a, xp, sp):
        ret = sp.linalg.eigsh(a, k=self.k, which=self.which,
                              return_eigenvectors=self.return_eigenvectors)
        if self.return_eigenvectors:
            w, x = ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, self.k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = self.res_tol[numpy.dtype(a.dtype).char.lower()]
            assert (res < tol)
        else:
            w = ret
        return xp.sort(w)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format == 'csc':
            pytest.xfail('may be buggy')  # trans=True

        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_eigsh(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_eigsh(a, xp, sp)

    degenerate_tol = {'f': 1e-5, 'd': 1e-10}

    @testing.for_dtypes('fdFD')
    def test_degenerate(self, dtype):
        # A degenerate spectrum exhausts the Krylov space (beta -> 0).
        # Without a breakdown guard this normalized by ~0 and returned NaN
        # (gh-6446, gh-7495). Check the identity yields all-ones, no NaN.
        if self.use_linear_operator:
            pytest.skip()
        a = sparse.identity(self.n, dtype=dtype, format='csr')
        w = sparse.linalg.eigsh(a, k=self.k, which=self.which,
                                return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        tol = self.degenerate_tol[numpy.dtype(dtype).char.lower()]
        cupy.testing.assert_allclose(
            cupy.sort(w), cupy.ones(self.k), rtol=tol, atol=tol)

    clustered_tol = {'f': 1e-4, 'd': 1e-6}

    @testing.for_dtypes('fdFD')
    def test_clustered_large_k(self, dtype):
        # A spectrum with only a few distinct eigenvalues exhausts its
        # Krylov space almost immediately (a 2-value spectrum has Krylov
        # dimension ~2): without the breakdown guard this returned NaN or
        # ghost/overflow Ritz values (~1e+217) at larger k (gh-7168,
        # gh-6769). The top-k eigenvalues are k exact copies of 50.
        if self.use_linear_operator or self.which != 'LA':
            pytest.skip()
        n, k = 200, 20
        evals = cupy.concatenate(
            [cupy.ones(n // 2, dtype='d'),
             50.0 * cupy.ones(n // 2, dtype='d')])
        q, _ = cupy.linalg.qr(testing.shaped_random((n, n), cupy,
                                                    dtype=dtype, seed=0))
        a = ((q * evals) @ q.conj().T).astype(dtype)
        w = sparse.linalg.eigsh(sparse.csr_matrix(a), k=k, which='LA',
                                return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        assert bool((cupy.abs(w) < 1e3).all())      # no ghost / overflow
        tol = self.clustered_tol[numpy.dtype(dtype).char.lower()]
        cupy.testing.assert_allclose(
            cupy.sort(w), cupy.full(k, 50.0), rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_clustered_large_k_gaussian(self, dtype):
        # The same two-value spectrum under a gaussian-QR rotation, with a
        # pinned start vector. Reported in review: this rotation returns
        # ghost Ritz values up to ~1e307 (NaN on stock) where the rotation
        # above happens to converge, because the restart locks duplicate
        # rows into the basis once an eigenvalue is degenerate. Both the
        # rotation and v0 are built on the host with seeded numpy, so the
        # input does not depend on GPU RNG.
        if self.use_linear_operator or self.which != 'LA':
            pytest.skip()
        n, k = 200, 20
        rng = numpy.random.default_rng(1)
        q, _ = numpy.linalg.qr(rng.standard_normal((n, n)))
        evals = numpy.concatenate(
            [numpy.ones(n // 2), 50.0 * numpy.ones(n // 2)])
        a = cupy.asarray((q * evals) @ q.T).astype(dtype)
        v0 = cupy.asarray(
            numpy.random.default_rng(1).random(n)).astype(dtype)
        w = sparse.linalg.eigsh(sparse.csr_matrix(a), k=k, which='LA',
                                v0=v0, return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        assert bool((cupy.abs(w) < 1e3).all())      # no ghost / overflow
        tol = self.clustered_tol[numpy.dtype(dtype).char.lower()]
        cupy.testing.assert_allclose(
            cupy.sort(w.real), cupy.full(k, 50.0), rtol=tol, atol=tol)

    @testing.for_dtypes('fdFD')
    def test_null_space_start(self, dtype):
        # v0 exactly in the null space of a singular A: beta and the norm
        # estimate both start at 0, so the breakdown test must be
        # non-strict (<=) for the reseed to fire -- with strict < the
        # whole Krylov space silently stays zero and eigsh returns zeros.
        if self.use_linear_operator or self.which != 'LA':
            pytest.skip()
        n = 10
        a = sparse.diags(cupy.arange(n).astype(dtype)).tocsr()
        v0 = cupy.zeros(n, dtype=dtype)
        v0[0] = 1                      # eigenvector of eigenvalue 0
        w = sparse.linalg.eigsh(a, k=2, which='LA', v0=v0,
                                return_eigenvectors=False)
        # with '<' this returned [0, 0]; with '<=' the reseeds explore true
        # eigendirections. (A dead v0 plus ncv = n - 1 leaves one dimension
        # unexplored and the exact-invariant res = 0 stop ends there, so the
        # result is genuine nonzero eigenvalues, not necessarily the
        # extremal pair -- inherited Lanczos semantics.)
        w = cupy.sort(w.real)
        assert not bool(cupy.isnan(w).any())
        assert float(w.min()) > 0.5
        cupy.testing.assert_allclose(w, cupy.around(w), atol=1e-4)

    @testing.for_dtypes('fdFD')
    def test_negative_semidefinite_la(self, dtype):
        # 'LA' on a negative-semidefinite operator with nullity >= k: the
        # zero eigenvalues ARE the largest algebraic targets, so the reseed
        # bias must not steer away from the null space -- 'LA' biases by
        # (A + anorm*I), which keeps the null space alive (see
        # _restart_ortho).
        if self.use_linear_operator or self.which != 'LA':
            pytest.skip()
        a = sparse.diags(cupy.concatenate(
            [cupy.zeros(5), -cupy.ones(5)]).astype(dtype)).tocsr()
        w = sparse.linalg.eigsh(a, k=5, which='LA',
                                return_eigenvectors=False)
        cupy.testing.assert_allclose(w.real, cupy.zeros(5), atol=1e-5)

    @testing.for_dtypes('fdFD')
    def test_semidefinite_la_null_not_targeted(self, dtype):
        # The mirror image of the test above: on a POSITIVE-semidefinite (or
        # indefinite) operator with a null space, zero is NOT an LA target,
        # so the reseed must keep annihilating null candidates rather than
        # spending Krylov slots on them. The shift is therefore applied only
        # when no positive Rayleigh quotient has been observed.
        if self.use_linear_operator or self.which != 'LA':
            pytest.skip()
        for vals in ([1.0, 2.0, 3.0] + [0.0] * 7,
                     [2.0, -3.0] + [0.0] * 8):
            a = sparse.diags(
                cupy.asarray(vals).astype(dtype)).tocsr()
            w = sparse.linalg.eigsh(a, k=3, which='LA',
                                    return_eigenvectors=False)
            ref = cupy.sort(cupy.asarray(vals).astype(
                cupy.asarray(vals).real.dtype))[-3:]
            cupy.testing.assert_allclose(
                cupy.sort(w.real), ref, atol=1e-4)

    # strict=False (pyproject sets xfail_strict): with the default v0 seeded
    # the outcome no longer varies run to run (every instance fails on an
    # A100), but the input B @ C is not symmetric, so what eigsh returns
    # for it is unspecified and the comparison itself is what gh-5001 has
    # to settle. Non-strict until the test is redefined.
    @pytest.mark.xfail(
        reason='eigsh works wrong (#5001)',
        raises=AssertionError,
        strict=False,
    )
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense_low_rank(self, dtype, xp, sp):
        n = self.n
        rank = 5
        # density is ignored.
        a = testing.shaped_random((n, rank), xp, dtype=dtype, scale=1).dot(
            testing.shaped_random((rank, n), xp, dtype=dtype, scale=1))
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_eigsh(a, xp, sp)

    def test_invalid(self):
        if self.use_linear_operator is True:
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = xp.diag(xp.ones((self.n, ), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(xp.ones((2, 1), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(a, k=0)
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones((self.n, ), dtype='f'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((1,), dtype='f'))
        with pytest.raises(TypeError):
            sp.linalg.eigsh(xp.ones((2, 2), dtype='i'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.n)
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.k, which='SM')

    def test_starting_vector(self):
        eigsh = cupyx.scipy.sparse.linalg.eigsh
        n = 100

        # Make symmetric matrix
        aux = cupy.random.randn(n, n)
        matrix = (aux + aux.T) / 2.0

        # Find reference eigenvector
        ew, ev = eigsh(matrix, k=1)
        v = ev[:, 0]

        # Obtain non-converged eigenvector from random initial guess.
        ew_aux, ev_aux = eigsh(matrix, k=1, ncv=1, maxiter=0)
        v_aux = cupy.copysign(ev_aux[:, 0], v)

        # Obtain eigenvector using known eigenvector as initial guess.
        ew_v0, ev_v0 = eigsh(matrix, k=1, v0=v.copy(), ncv=1, maxiter=0)
        v_v0 = cupy.copysign(ev_v0[:, 0], v)

        assert cupy.linalg.norm(v - v_v0) < cupy.linalg.norm(v - v_aux)


@testing.with_requires('scipy')
class TestEigshLateNormDiscovery:
    # An operator with wide dynamic range and a v0 orthogonal to its
    # dominant eigenspace -- what a deflation workflow produces -- makes
    # the first sweep honestly underestimate ||A||, and a later reseed
    # legitimately discovers it. A divergence guard measured against the
    # first sweep's own estimate mistakes that for corruption; one
    # measured against a true bound on ||A|| cannot.
    @testing.for_dtypes('fdFD')
    @pytest.mark.parametrize('big', [1e4, 1e6])
    def test_dominant_eigenvalue_hidden_from_v0(self, dtype, big):
        n = 400
        d = numpy.ones(n)
        d[0] = big
        a = sparse.diags(cupy.asarray(d.astype(
            numpy.dtype(dtype).char.lower()))).tocsr().astype(dtype)
        v0 = numpy.random.default_rng(0).random(n)
        v0[0] = 0.0                        # exactly orthogonal to e_0
        v0 = cupy.asarray((v0 / numpy.linalg.norm(v0)).astype(dtype))
        w = sparse.linalg.eigsh(a, k=6, which='LA', v0=v0,
                                return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        assert bool((cupy.abs(w) <= 8 * big).all())


@testing.with_requires('scipy')
class TestEigshTinyN:
    # n = 2 forces ncv = n - 1 = 1, so the sweep yields a single row and the
    # breakdown walk has no interior beta to inspect. Regression guard: the
    # running-max array is empty there, and indexing it raised IndexError
    # before the first Ritz solve. With v0 an exact eigenvector the first
    # sweep converges (res = 0), which is the path that must keep working;
    # a non-converging n = 2 input still hits the separate V[k+1] bound
    # issue tracked in #10220.
    @testing.for_dtypes('fdFD')
    def test_n2_exact_v0(self, dtype):
        a = cupy.array([[2, 0], [0, 1]], dtype=dtype)
        v0 = cupy.array([1, 0], dtype=dtype)
        w = sparse.linalg.eigsh(sparse.csr_matrix(a), k=1, which='LM',
                                v0=v0, return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        cupy.testing.assert_allclose(w.real, cupy.array([2.0]),
                                     rtol=1e-5, atol=1e-5)


@testing.parameterize(*testing.product({
    'which': ['LM', 'SA'],
    'shift': [0.0, 0.5],
}))
@testing.with_requires('scipy')
class TestEigshDegenerateHermitian:
    # Hermitian, well-posed, and silently wrong on stock: B @ B^H with
    # rank 5 << n has an eigenvalue of multiplicity n - 5, so the Krylov
    # space is exhausted after ~5 steps and stock normalizes by ~0 --
    # returning NaN, or ghost Ritz values, depending on the draw. The
    # +0.5*I variant moves the degenerate block off zero, so a failure
    # cannot be dismissed as a null-space artifact.
    #
    # The exact spectrum is known analytically, which matters here: a
    # dense eigvalsh of the n x n matrix is LESS accurate than eigsh on
    # the degenerate block (measured 1.9e-3 vs 3e-4 in float32), so it is
    # not a usable reference. B^H B is only rank x rank and carries the
    # nonzero eigenvalues exactly; everything else is the shift.
    n = 100
    rank = 5
    k = 6

    @testing.for_dtypes('fdFD')
    def test_low_rank_gram(self, dtype):
        b = testing.shaped_random((self.n, self.rank), cupy, dtype=dtype,
                                  seed=0)
        a = b @ b.conj().T
        a = (a + a.conj().T) / 2              # exactly Hermitian
        if self.shift:
            a = a + self.shift * cupy.eye(self.n, dtype=dtype)

        # Exact spectrum: rank nonzero eigenvalues from the small Gram,
        # the rest are the shift (multiplicity n - rank).
        nonzero = cupy.linalg.eigvalsh(b.conj().T @ b) + self.shift
        if self.which == 'LM':
            expected = cupy.sort(cupy.concatenate(
                [nonzero, cupy.asarray([self.shift], dtype=nonzero.dtype)]))
        else:                                  # 'SA': the degenerate block
            expected = cupy.full((self.k,), self.shift,
                                 dtype=nonzero.dtype)

        w = sparse.linalg.eigsh(sparse.csr_matrix(a), k=self.k,
                                which=self.which,
                                return_eigenvectors=False)
        assert not bool(cupy.isnan(w).any())
        assert bool(cupy.isfinite(w).all())
        # Absolute tolerance must scale with ||A||: the degenerate block
        # sits at the roundoff floor of the largest eigenvalue, not at an
        # absolute one. Same 64*eps*||A|| scale the solver itself uses.
        anorm = float(cupy.abs(nonzero).max())
        eps = float(numpy.finfo(numpy.dtype(dtype).char.lower()).eps)
        cupy.testing.assert_allclose(
            cupy.sort(w.real), cupy.sort(expected.real),
            rtol=1e-4, atol=64 * eps * anorm)


@testing.with_requires('scipy')
class TestSvdsV0:
    """v0= passthrough (scipy-compatible): reproducibility + parity."""

    def _mat(self, xp, m=60, n=45):
        return testing.shaped_random((m, n), xp, dtype='d', scale=1)

    def test_v0_deterministic(self):
        # v0 pins the trajectory start, which removes the run-to-run TIME
        # variance; it is NOT bitwise: the adjoint half of the Gram apply
        # is a transpose-mode cuSPARSE SpMV whose default algorithm uses
        # atomics, so accumulation order (and the last bits of the
        # result) still varies between identical calls.
        a = sparse.csr_matrix(self._mat(cupy))
        v0 = testing.shaped_random((45,), cupy, dtype='d', scale=1, seed=7)
        v0_in = v0.copy()
        s1 = sparse.linalg.svds(a, k=5, v0=v0,
                                return_singular_vectors=False)
        s2 = sparse.linalg.svds(a, k=5, v0=v0,
                                return_singular_vectors=False)
        cupy.testing.assert_allclose(cupy.sort(s1), cupy.sort(s2),
                                     rtol=1e-9, atol=1e-9)
        # v0 is an input, not a workspace: the caller's array is unchanged.
        cupy.testing.assert_array_equal(v0, v0_in)

    def test_v0_matches_scipy(self):
        import numpy
        import scipy.sparse
        import scipy.sparse.linalg
        a_np = testing.shaped_random((60, 45), numpy, dtype='d', scale=1)
        v0_np = testing.shaped_random((45,), numpy, dtype='d', scale=1,
                                      seed=7)
        s_sp = numpy.sort(scipy.sparse.linalg.svds(
            scipy.sparse.csr_matrix(a_np), k=5, v0=v0_np,
            return_singular_vectors=False))
        s_cp = cupy.sort(sparse.linalg.svds(
            sparse.csr_matrix(cupy.asarray(a_np)), k=5,
            v0=cupy.asarray(v0_np), return_singular_vectors=False))
        numpy.testing.assert_allclose(cupy.asnumpy(s_cp), s_sp, rtol=1e-8,
                                      atol=1e-8)

    def test_v0_wide_matrix_length(self):
        # v0 length is min(a.shape) regardless of orientation
        a = sparse.csr_matrix(self._mat(cupy, m=45, n=60))
        v0 = testing.shaped_random((45,), cupy, dtype='d', scale=1, seed=3)
        s = sparse.linalg.svds(a, k=5, v0=v0,
                               return_singular_vectors=False)
        assert not bool(cupy.isnan(s).any())


class TestDefaultStartVector:
    # The default start vector comes from a fixed seed, so a call with
    # v0=None is reproducible and the global cupy.random state cannot leak
    # into eigsh/svds results (gh-10239: the same test instance passed or
    # failed depending on the draw).

    @testing.for_dtypes('fdFD')
    def test_default_v0_is_fixed(self, dtype):
        from cupyx.scipy.sparse.linalg import _eigen
        u1 = _eigen._default_v0(1000, dtype)
        cupy.random.seed(1)
        cupy.random.random(1000)         # advance the global generator
        u2 = _eigen._default_v0(1000, dtype)
        assert u1.dtype == cupy.dtype(dtype)
        cupy.testing.assert_array_equal(u1, u2)
        assert float(cupy.abs(u1 - u1.mean()).max()) > 0.1  # not constant

    @testing.for_dtypes('fdFD')
    def test_eigsh_repeatable(self, dtype):
        b = testing.shaped_random((120, 120), cupy, dtype=dtype, seed=0)
        a = sparse.csr_matrix(b + b.conj().T)
        w1 = sparse.linalg.eigsh(a, k=6, return_eigenvectors=False)
        cupy.random.seed(2)
        w2 = sparse.linalg.eigsh(a, k=6, return_eigenvectors=False)
        # Same start, same trajectory: only the parallel-reduction order in
        # cuBLAS/cuSPARSE differs between the two runs.
        tol = 1e-5 if numpy.dtype(dtype).char in 'fF' else 1e-10
        cupy.testing.assert_allclose(cupy.sort(w1.real), cupy.sort(w2.real),
                                     rtol=tol, atol=0)

    def test_svds_augmented_vectors_repeatable(self):
        # k above the rank: the missing singular vectors are completed by
        # random orthonormal columns, which must be reproducible as well.
        m, n, rank = 40, 30, 3
        a = (testing.shaped_random((m, rank), cupy, dtype='d', seed=0)
             @ testing.shaped_random((rank, n), cupy, dtype='d', seed=1))
        u1, s1, vt1 = sparse.linalg.svds(sparse.csr_matrix(a), k=6)
        cupy.random.seed(3)
        u2, s2, vt2 = sparse.linalg.svds(sparse.csr_matrix(a), k=6)
        cupy.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-10)
        cupy.testing.assert_allclose(u1, u2, rtol=1e-8, atol=1e-8)
        cupy.testing.assert_allclose(vt1, vt2, rtol=1e-8, atol=1e-8)


@testing.parameterize(*testing.product({
    'shape': [(30, 29), (29, 29), (29, 30)],
    'k': [3, 6, 12],
    'return_vectors': [True, False],
    'use_linear_operator': [True, False],
}))
@testing.with_requires('scipy')
class TestSvds:
    density = 0.33
    tol = {numpy.float32: 1e-4, numpy.complex64: 2e-4, 'default': 1e-12}

    def _make_matrix(self, dtype, xp):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        mask = testing.shaped_random(self.shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        return a

    def _test_svds(self, a, xp, sp):
        ret = sp.linalg.svds(a, k=self.k,
                             return_singular_vectors=self.return_vectors)
        if self.return_vectors:
            u, s, vt = ret
            # Check the results with u @ s @ vt, as singular vectors don't
            # necessarily match.
            return u @ xp.diag(s) @ vt
        else:
            return xp.sort(ret)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format in ('csr', 'csc'):
            pytest.xfail('may be buggy')  # trans=True

        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_svds(a, xp, sp)

    @testing.for_dtypes('fdFD')
    def test_rank_deficient(self, dtype):
        # A rank-deficient matrix gives A^H A a large null space, exhausting
        # the Krylov space; without a breakdown guard svds collapsed to all
        # zeros (gh-8009). Check it recovers the true top-k values: to
        # machine precision in d/D, and to ~1% in f/F GIVEN AN ACHIEVABLE
        # tol -- the default tol = eps is an absolute residual that single
        # precision cannot reach for a matrix of this norm, so the solve
        # would run to maxiter and return a partially converged tail.
        if self.use_linear_operator:
            pytest.skip()
        m, n = self.shape
        rank = min(m, n) // 2
        a = (testing.shaped_random((m, rank), cupy, dtype=dtype, seed=0)
             @ testing.shaped_random((rank, n), cupy, dtype=dtype, seed=1))
        if numpy.dtype(dtype).char.lower() == 'd':
            svds_tol, cmp_tol = 0, 1e-4
        else:
            svds_tol, cmp_tol = 1e-6 * float(cupy.linalg.norm(a)) ** 2, 5e-2
        s = sparse.linalg.svds(sparse.csr_matrix(a), k=self.k, tol=svds_tol,
                               return_singular_vectors=False)
        assert not bool(cupy.isnan(s).any())
        ref = cupy.sort(cupy.linalg.svd(a, compute_uv=False)[:self.k])
        cupy.testing.assert_allclose(
            cupy.sort(s), ref, rtol=cmp_tol, atol=cmp_tol * float(ref.max()))

    low_rank_wide_tol = {'f': 1e-2, 'd': 1e-6}

    @testing.for_dtypes('fdFD')
    def test_low_rank_wide(self, dtype):
        # gh-8009's exact regime: rank 5 in a 100x1000 matrix, so A^H A has
        # a 95% null space. The breakdown reseed must be biased out of the
        # null space (one application of the operator, see _restart_ortho)
        # -- an unbiased canonical reseed wastes the Krylov slots on
        # eigenvalue-0 directions and default ncv recovers only 1-3 of the
        # 5 true values. Runs once (not per class parameter).
        if (self.use_linear_operator or self.return_vectors
                or self.shape != (30, 29) or self.k != 6):
            pytest.skip()
        rank = 5
        a = (testing.shaped_random((100, rank), cupy, dtype=dtype, seed=0)
             @ testing.shaped_random((rank, 1000), cupy, dtype=dtype,
                                     seed=1))
        s = cupy.sort(sparse.linalg.svds(sparse.csr_matrix(a), k=6,
                      return_singular_vectors=False))
        assert not bool(cupy.isnan(s).any())
        ref = cupy.linalg.svd(a, compute_uv=False)[:rank]
        tol = self.low_rank_wide_tol[numpy.dtype(dtype).char.lower()]
        cupy.testing.assert_allclose(
            cupy.sort(s[1:]), cupy.sort(ref), rtol=tol,
            atol=tol * float(ref.max()))
        assert float(s[0]) < 1e-3 * float(ref.max())   # the rank-6 value ~ 0

    def test_low_rank_coordinate_null(self):
        # gh-8009's literal reproducer: eye(100, 1000) with rows 5+ zeroed
        # makes A^H A a COORDINATE projector, so every canonical reseed
        # lands exactly in the null space (bias_op @ e_j == 0); the varied
        # dense probe in _restart_ortho is what recovers all five values.
        if (self.use_linear_operator or self.return_vectors
                or self.shape != (30, 29) or self.k != 6):
            pytest.skip()
        a = cupy.eye(100, 1000, dtype='d')
        a[5:] = 0
        s = cupy.sort(sparse.linalg.svds(sparse.csr_matrix(a), k=6,
                      return_singular_vectors=False))
        assert not bool(cupy.isnan(s).any())
        cupy.testing.assert_allclose(s[1:], cupy.ones(5), atol=1e-8)
        assert float(s[0]) < 1e-6

    # strict=False (pyproject sets xfail_strict): with the default v0 seeded
    # the split is deterministic -- on an A100 the 26 instances without
    # singular vectors or with m < n pass and the 10 with vectors and
    # m >= n fail, identically across runs -- so this is gh-5001 proper,
    # not the start vector. Non-strict until the split is confirmed on
    # CI hardware; the passing instances can then be asserted.
    @pytest.mark.xfail(
        reason='eigsh works wrong (#5001)',
        raises=AssertionError,
        strict=False,
    )
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=tol, atol=tol, sp_name='sp')
    def test_dense_low_rank(self, dtype, xp, sp):
        m, n = self.shape
        rank = 5
        # density is ignored.
        a = testing.shaped_random((m, rank), xp, dtype=dtype, scale=1).dot(
            testing.shaped_random((rank, n), xp, dtype=dtype, scale=1))
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_svds(a, xp, sp)

    def test_invalid(self):
        if self.use_linear_operator is True:
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = xp.diag(xp.ones(self.shape, dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.svds(a, k=0)
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones(self.shape, dtype='f'))
        with pytest.raises(ValueError):
            sp.linalg.svds(xp.ones((1,), dtype='f'))
        with pytest.raises(TypeError):
            sp.linalg.svds(xp.ones((2, 2), dtype='i'))
        with pytest.raises(ValueError):
            sp.linalg.svds(a, k=min(self.shape))
        with pytest.raises(ValueError):
            sp.linalg.svds(a, k=self.k, which='SM')


@testing.parameterize(*testing.product({
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestCg:
    n = 30
    density = 0.33
    _atol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, 10)
        a = testing.shaped_random(shape, xp, dtype=dtype.char.lower(), scale=1)
        if dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(
                shape, xp, dtype=dtype.char.lower(), scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a @ a.conj().T
        a = a + xp.diag(xp.ones((self.n,), dtype=dtype.char.lower()))
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype)
        return b / xp.linalg.norm(b)

    def _test_cg(self, dtype, xp, sp, a, M):
        dtype = numpy.dtype(dtype)
        b = self._make_normalized_vector(dtype, xp)
        if self.b_ndim == 2:
            b = b.reshape(self.n, 1)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,), dtype=dtype)
        atol = 0.0
        if self.atol == 'select-by-dtype':
            atol = self._atol[dtype.char.lower()]
        return sp.linalg.cg(a, b, x0=x0, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format == 'csc':
            pytest.xfail('may be buggy')  # trans=True

        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cg(dtype, xp, sp, a, M)

    @testing.with_requires('scipy')
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        if self.atol is None and xp == numpy:
            return sp.linalg.cg(a, b)
        else:
            return sp.linalg.cg(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.cg(a, b, callback=callback)
        assert is_called

    def test_invalid(self):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, x0=ng_x0, atol=self.atol)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, M=ng_M, atol=self.atol)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.cg(ng_a, b, atol=self.atol)


@testing.parameterize(*testing.product({
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestBicgstab:
    n = 30
    density = 0.33
    _atol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, 10)
        a = testing.shaped_random(shape, xp, dtype=dtype.char.lower(), scale=1)
        if dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(
                shape, xp, dtype=dtype.char.lower(), scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a @ a.conj().T
        a = a + xp.diag(xp.ones((self.n,), dtype=dtype.char.lower()))
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype)
        return b / xp.linalg.norm(b)

    def _test_bicgstab(self, dtype, xp, sp, a, M):
        dtype = numpy.dtype(dtype)
        b = self._make_normalized_vector(dtype, xp)
        if self.b_ndim == 2:
            b = b.reshape(self.n, 1)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,), dtype=dtype)
        atol = 0.0
        if self.atol == 'select-by-dtype':
            atol = self._atol[dtype.char.lower()]
        return sp.linalg.bicgstab(a, b, x0=x0, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_bicgstab(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format == 'csc':
            pytest.xfail('may be buggy')  # trans=True

        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_bicgstab(dtype, xp, sp, a, M)

    @testing.with_requires('scipy')
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        return sp.linalg.bicgstab(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.bicgstab(a, b, callback=callback)
        assert is_called

    def test_invalid(self):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(ng_a, b, atol=self.atol)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(a, ng_b, atol=self.atol)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(a, ng_b, atol=self.atol)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(a, b, x0=ng_x0, atol=self.atol)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.bicgstab(a, b, M=ng_M, atol=self.atol)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.bicgstab(ng_a, b, atol=self.atol)


@testing.parameterize(*testing.product({
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
    'restart': [None, 10],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestGmres:
    n = 30
    density = 0.2
    _atol = {'f': 1e-5, 'd': 1e-12}

    # TODO(kataoka): Fix the `lstsq` call in CuPy's `gmres`
    @pytest.fixture(autouse=True)
    def ignore_futurewarning(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', '`rcond` parameter will change', FutureWarning,
            )
            yield

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype, scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        diag = xp.diag(testing.shaped_random(
            (self.n,), xp, dtype=dtype.char.lower(), scale=1) + 1)
        a[diag > 0] = 0
        a = a + diag
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype, scale=1)
        return b / xp.linalg.norm(b)

    def _test_gmres(self, dtype, xp, sp, a, M):
        dtype = numpy.dtype(dtype)
        b = self._make_normalized_vector(dtype, xp)
        if self.b_ndim == 2:
            b = b.reshape(self.n, 1)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,), dtype=dtype)
        atol = 0.0
        if self.atol == 'select-by-dtype':
            atol = self._atol[dtype.char.lower()]
        return sp.linalg.gmres(
            a, b, x0=x0, restart=self.restart, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_gmres(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format == 'csc':
            pytest.xfail('may be buggy')  # trans=True

        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_gmres(dtype, xp, sp, a, M)

    @testing.with_requires('scipy')
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            pytest.skip()
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        if self.atol is None and xp == numpy:
            return sp.linalg.gmres(a, b)
        else:
            return sp.linalg.gmres(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            pytest.skip()
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback1(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.gmres(a, b, callback=callback1, callback_type='x')
        assert is_called
        is_called = False

        def callback2(pr_norm):
            print(pr_norm)
            nonlocal is_called
            is_called = True
        sp.linalg.gmres(a, b, callback=callback2, callback_type='pr_norm')
        assert is_called

    def test_invalid(self):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.restart is None and self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, x0=ng_x0)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, M=ng_M)
            ng_callback_type = '?'
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, callback_type=ng_callback_type)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.gmres(ng_a, b)


@testing.with_requires('scipy')
class TestGmresInfo:

    def test_nonconvergence_with_restart_maxiter_mismatch(self):
        n = 48
        indices = numpy.arange(n, dtype=numpy.float64)
        a_cpu = 1.0 / (indices[:, None] + indices[None, :] + 1.0)
        a = sparse.csr_matrix(cupy.asarray(a_cpu))
        x_true = cupy.asarray(numpy.random.default_rng(0).standard_normal(n))
        b = a @ x_true
        tol = 1e-12

        x, info = sparse.linalg.gmres(
            a, b, restart=2, maxiter=3, atol=tol, rtol=tol)
        rel_res = cupy.linalg.norm(a @ x - b) / cupy.linalg.norm(b)

        assert rel_res > tol
        assert info != 0


def skip_HIP_spMM_error(outer=()):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if (runtime.is_hip and self.inner_modification == 'sparse'
                    and self.outer_modification in outer):
                pytest.xfail('spMM is buggy')  # trans=True
            impl(self, *args, **kw)
        return test_func
    return decorator


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'outer_modification': [
        'normal', 'transpose', 'hermitian'],
    'inner_modification': [
        'normal', 'sparse', 'linear_operator', 'class_matvec', 'class_matmat'],
    'M': [1, 6],
    'N': [1, 7],
}))
@testing.with_requires('scipy')
class TestLinearOperator:

    # modified from scipy
    # class that defines parametrized custom cases
    # adapted from scipy's analogous tests
    def _inner_cases(self, xp, sp, A):
        # creating base-matrix-like class with default
        # matrix-vector and adjoint-matrix-vector impl

        def mv(x):
            return A.dot(x)

        def rmv(x):
            return A.T.conj().dot(x)

        # defining the user-defined classes
        class BaseMatlike(sp.linalg.LinearOperator):

            def __init__(self):
                self.dtype = A.dtype
                self.shape = A.shape

            def _adjoint(self):
                shape = self.shape[1], self.shape[0]
                return sp.linalg.LinearOperator(
                    matvec=rmv, rmatvec=mv, dtype=self.dtype, shape=shape)

        class HasMatvec(BaseMatlike):

            def _matvec(self, x):
                return mv(x)

        class HasMatmat(BaseMatlike):

            def _matmat(self, x):
                return mv(x)

        if self.inner_modification == 'normal':
            return sp.linalg.aslinearoperator(A)
        if self.inner_modification == 'sparse':
            # TODO(asi1024): Fix to return contiguous matrix.
            return sp.linalg.aslinearoperator(sp.csr_matrix(A))
        if self.inner_modification == 'linear_operator':
            return sp.linalg.LinearOperator(
                matvec=mv, rmatvec=rmv, dtype=A.dtype, shape=A.shape)
        if self.inner_modification == 'class_matvec':
            return HasMatvec()
        if self.inner_modification == 'class_matmat':
            return HasMatmat()
        assert False

    def _generate_linear_operator(self, xp, sp):
        A = testing.shaped_random((self.M, self.N), xp, self.dtype)

        if self.outer_modification == 'normal':
            return self._inner_cases(xp, sp, A)
        if self.outer_modification == 'transpose':
            # From SciPy 1.4 (scipy/scipy#9064)
            return self._inner_cases(xp, sp, A.T).T
        if self.outer_modification == 'hermitian':
            return self._inner_cases(xp, sp, A.T.conj()).H
        assert False

    @skip_HIP_spMM_error(outer=('transpose', 'hermitian'))
    @testing.numpy_cupy_allclose(sp_name='sp', rtol=1e-6)
    def test_matvec(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x_1dim = testing.shaped_random((self.N,), xp, self.dtype)
        x_2dim = testing.shaped_random((self.N, 1), xp, self.dtype)
        return linop.matvec(x_1dim), linop.matvec(x_2dim)

    @skip_HIP_spMM_error(outer=('transpose', 'hermitian'))
    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_matmat(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop.matmat(x)

    @skip_HIP_spMM_error(outer=('normal',))
    @testing.numpy_cupy_allclose(sp_name='sp', rtol=1e-6)
    def test_rmatvec(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x_1dim = testing.shaped_random((self.M,), xp, self.dtype)
        x_2dim = testing.shaped_random((self.M, 1), xp, self.dtype)
        return linop.rmatvec(x_1dim), linop.rmatvec(x_2dim)

    @skip_HIP_spMM_error(outer=('normal',))
    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_rmatmat(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x = testing.shaped_random((self.M, 8), xp, self.dtype)
        return linop.rmatmat(x)

    @skip_HIP_spMM_error(outer=('transpose', 'hermitian'))
    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_dot(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x0 = testing.shaped_random((self.N,), xp, self.dtype)
        x1 = testing.shaped_random((self.N, 1), xp, self.dtype)
        x2 = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop.dot(x0), linop.dot(x1), linop.dot(x2)

    @skip_HIP_spMM_error(outer=('transpose', 'hermitian'))
    @testing.numpy_cupy_allclose(
        sp_name='sp', rtol=1e-6, contiguous_check=False)
    def test_mul(self, xp, sp):
        linop = self._generate_linear_operator(xp, sp)
        x0 = testing.shaped_random((self.N,), xp, self.dtype)
        x1 = testing.shaped_random((self.N, 1), xp, self.dtype)
        x2 = testing.shaped_random((self.N, 8), xp, self.dtype)
        return linop * x0, linop * x1, linop * x2


@testing.parameterize(*testing.product({
    'lower': [True, False],
    'unit_diagonal': [True, False],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F']
}))
@testing.with_requires('scipy')
@pytest.mark.skipif(not cusparse.check_availability('csrsm2'),
                    reason='no working implementation')
class TestSpsolveTriangular:

    n = 10
    density = 0.5

    def _make_matrix(self, dtype, xp):
        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, xp, dtype=dtype, scale=1)
        mask = testing.shaped_random(a_shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        diag = xp.diag(xp.ones((self.n,), dtype=dtype))
        a = a + diag
        if self.lower:
            a = xp.tril(a)
        else:
            a = xp.triu(a)
        b_shape = (self.n,) if self.nrhs is None else (self.n, self.nrhs)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, order=self.order)
        return a, b

    def _test_spsolve_triangular(self, sp, a, b):
        return sp.linalg.spsolve_triangular(a, b, lower=self.lower,
                                            unit_diagonal=self.unit_diagonal)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(
        rtol=1e-5, atol=1e-5, sp_name='sp', contiguous_check=False,
        type_check=False,  # "XXX: Dtypes differ on np2.0 / win scipy1.14
    )
    def test_sparse(self, format, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        return self._test_spsolve_triangular(sp, a, b)

    def test_invalid_cases(self):
        dtype = 'float64'
        if not (self.lower and self.unit_diagonal and self.nrhs == 4 and
                self.order == 'C'):
            pytest.skip()

        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, b = self._make_matrix(dtype, xp)
            a = sp.csr_matrix(a)

            # a is not a square matrix
            ng_a = sp.csr_matrix(xp.ones((self.n + 1, self.n), dtype=dtype))
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, ng_a, b)
            # b is not a 1D/2D matrix
            ng_b = xp.ones((1, self.n, self.nrhs), dtype=dtype)
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, a, ng_b)
            # mismatched shape
            ng_b = xp.ones((self.n + 1, self.nrhs), dtype=dtype)
            with pytest.raises(ValueError):
                self._test_spsolve_triangular(sp, a, ng_b)

        xp, sp = cupy, sparse
        a, b = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)

        # unsupported dtype
        ng_a = sp.csr_matrix(xp.ones((self.n, self.n), dtype='bool'))
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, ng_a, b)
        # a is not spmatrix
        ng_a = xp.ones((self.n, self.n), dtype=dtype)
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, ng_a, b)
        # b is not cupy ndarray
        ng_b = numpy.ones((self.n, self.nrhs), dtype=dtype)
        with pytest.raises(TypeError):
            self._test_spsolve_triangular(sp, a, ng_b)


@testing.parameterize(*testing.product({
    'tol': [0, 1e-5],
    'reorder': [0, 1, 2, 3],
}))
@testing.with_requires('scipy')
@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
class TestCsrlsvqr:

    n = 8
    density = 0.75
    _test_tol = {'f': 1e-5, 'd': 1e-12}

    def _setup(self, dtype):
        dtype = numpy.dtype(dtype)
        a_shape = (self.n, self.n)
        a = testing.shaped_random(
            a_shape, numpy, dtype=dtype, scale=2 / self.n)
        a_mask = testing.shaped_random(a_shape, numpy, dtype='f', scale=1)
        a[a_mask > self.density] = 0
        a_diag = numpy.diag(numpy.ones((self.n,), dtype=dtype))
        a = a + a_diag
        b = testing.shaped_random((self.n,), numpy, dtype=dtype)
        test_tol = self._test_tol[dtype.char.lower()]
        return a, b, test_tol

    @testing.for_dtypes('fdFD')
    def test_csrlsvqr(self, dtype):
        a, b, test_tol = self._setup(dtype)
        ref_x = numpy.linalg.solve(a, b)
        cp_a = cupy.array(a)
        sp_a = cupyx.scipy.sparse.csr_matrix(cp_a)
        cp_b = cupy.array(b)
        x = cusolver.csrlsvqr(sp_a, cp_b, tol=self.tol,
                              reorder=self.reorder)
        cupy.testing.assert_allclose(x, ref_x, rtol=test_tol,
                                     atol=test_tol)


@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
@testing.with_requires('scipy')
class TestSpSolve:
    def _check_spsolve(self, xp, sp, dtyp):
        n, nb = 5, 3

        a = xp.diag(xp.arange(n) + 1)
        sa = sp.csr_matrix(a.astype(dtyp))

        # prepare b to be non-contiguous
        b = xp.arange((2*n*nb), dtype=dtyp).reshape((2*n, nb))
        b = b[::2, :]
        result = sp.linalg.spsolve(sa, b)
        return result

    @pytest.mark.parametrize('dtyp', ['float32', 'complex64'])
    @testing.numpy_cupy_allclose(sp_name='sp', atol=5e-7)
    def test_spsolve_single(self, xp, sp, dtyp):
        return self._check_spsolve(xp, sp, dtyp)

    @pytest.mark.parametrize('dtyp', ['float64', 'complex128'])
    @testing.numpy_cupy_allclose(sp_name='sp', atol=1e-14)
    def test_spsolve_double(self, xp, sp, dtyp):
        return self._check_spsolve(xp, sp, dtyp)


def _eigen_vec_transform(block_vec, xp):
    """Helper to swap sign of each eigen vector based on the first
    non-zero element. ie, to standardize the first non-zero element
    of eigen vector as positive. This helps in comparing equivalence
    of eigen vectors"""
    direction = testing.shaped_random((block_vec.shape[0], 1),
                                      xp=xp, seed=123)
    direction = xp.where(block_vec.T.dot(direction) >= 0, 1, -1).T
    # shape of mask: (block_vec.shape[0], 1)
    # each eigenvector is multiplied by a 1 or -1 (scalar)
    # this is done by broadcasting mask
    return block_vec * direction


@testing.with_requires('scipy')
@pytest.mark.skipif(runtime.is_hip and driver.get_build_version() < 402,
                    reason='syevj not available')
# tests adapted from scipy's tests of lobpcg
class TestLOBPCG:

    def _generate_input_for_elastic_rod(self, n, xp):
        """Build the matrices for the generalized eigenvalue problem of the
        fixed-free elastic rod vibration model.
        """
        L = 1.0
        le = L / n
        rho = 7.85e3
        S = 1.e-4
        E = 2.1e11
        mass = rho * S * le / 6.
        k = E * S / le
        A = k * (xp.diag(xp.r_[2. * xp.ones(n - 1), 1]) -
                 xp.diag(xp.ones(n - 1), 1) - xp.diag(xp.ones(n - 1), -1))
        B = mass * (xp.diag(xp.r_[4. * xp.ones(n - 1), 2]) +
                    xp.diag(xp.ones(n - 1), 1) + xp.diag(xp.ones(n - 1), -1))
        return A, B

    def _generate_input_for_mikota_pair(self, n, xp):
        """Build a pair of full diagonal matrices for the generalized eigenvalue
        problem. The Mikota pair acts as a nice test since the eigenvalues are
        the squares of the integers n, n=1,2,...
        """  # NOQA
        x = xp.arange(1, n + 1)
        B = xp.diag(1. / x)
        y = xp.arange(n - 1, 0, -1)
        z = xp.arange(2 * n - 1, 0, -2)
        A = xp.diag(z) - xp.diag(y, -1) - xp.diag(y, 1)
        return A, B

    def _generate_random_initial_ortho_eigvec(self, m, n, xp=numpy, seed=0):
        """helper to generate orthogonal, random initial approximation for
        eigen vectors.
        """
        V = testing.shaped_random((m, n), xp=numpy, seed=seed)
        # TODO : use cupy's native linalg.orth() once implemented
        X = scipy.linalg.orth(V)
        return xp.asarray(X)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_small_generate_input_for_elastic_rod(self, xp, sp):
        A, B = self._generate_input_for_elastic_rod(10, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 10, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_small_generate_input_for_mikota_pair(self, xp, sp):
        A, B = self._generate_input_for_mikota_pair(10, xp)
        n = A.shape[0]
        X = self._generate_random_initial_ortho_eigvec(n, 10, xp)
        eigvals, eigvecs = sp.linalg.lobpcg(A,
                                            X, B=B,
                                            tol=1e-5, maxiter=30,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_regression(self, xp, sp):
        """Check the eigenvalue of the identity matrix is one.
        """
        n = 10
        X = xp.ones((n, 1))
        A = xp.identity(n)
        w, v = sp.linalg.lobpcg(A, X)
        return w, _eigen_vec_transform(v, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_diagonal(self, xp, sp):
        """Check for diagonal matrices.
        """
        # The system of interest is of size n x n.
        n = 100
        # We care about only m eigenpairs.
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # We choose A to be the diagonal matrix whose entries are 1..n
        # and where B is chosen to be the identity matrix.
        vals = xp.arange(1, n + 1, dtype=float)
        A = sp.diags([vals], [0], (n, n))
        B = sp.eye(n)
        # Let the preconditioner M be the inverse of A.
        M = sp.diags([1. / vals], [0], (n, n))
        # Pick random initial vectors.
        X = testing.shaped_random((n, m), xp=xp, seed=1234)
        # Require that the returned eigenvectors be in the orthogonal
        # complement of the first few standard basis vectors (Y)
        m_excluded = 3
        Y = xp.eye(n, m_excluded)
        eigvals, vecs = sp.linalg.lobpcg(A, X, B, M=M, Y=Y, tol=1e-4,
                                         maxiter=40, largest=False)
        return eigvals, _eigen_vec_transform(vecs, xp)

    def _generate_A_for_fiedler(self, n, p, xp):
        """Check for fiedler vector computation"""
        # fiedler vector computation based on scipy's tests
        # https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/sparse/linalg/eigen/lobpcg/tests/test_lobpcg.py#L140
        col = numpy.zeros(n)
        col[1] = 1
        A = scipy.linalg.toeplitz(col)
        D = numpy.diag(A.sum(axis=1))
        return xp.asarray(D - A)

    def _generate_small_X_for_fiedler(self, n, p, xp):
        tmp = xp.pi * xp.arange(n) / n
        analytic_V = xp.cos(xp.outer(xp.arange(n) + 1 / 2, tmp))
        return analytic_V[:, :p]

    def _generate_large_X_for_fiedler(self, n, p, xp):
        tmp = xp.pi * xp.arange(n) / n
        analytic_V = xp.cos(xp.outer(xp.arange(n) + 1 / 2, tmp))
        return analytic_V[:, -p:]

    def _generate_approximate_X_for_fiedler(self, n, p, xp):
        fiedler_guess = xp.concatenate((xp.ones(n // 2),
                                        -xp.ones(n - n // 2)))
        return xp.vstack((xp.ones(n), fiedler_guess)).T

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_small_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for small fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_small_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_large_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for large fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_large_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_approximate_8(self, xp, sp):
        """Check the dense workaround path for small matrices
           for approximately-formed fiedler eigen values and vectors
        """
        # This triggers the dense path because 8 < 2*5.
        A = self._generate_A_for_fiedler(8, 2, xp)
        X = self._generate_approximate_X_for_fiedler(8, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_small_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small
           fiedler matrices and small eigen values and vectors
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_small_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_large_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small
           fiedler matrices and large eigen values and vectors
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_large_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    def test_fiedler_approximate_12(self, xp, sp):
        """Check the dense workaround path is avoided for non-small,
           approximately generated fiedler matrices
        """
        A = self._generate_A_for_fiedler(12, 2, xp)
        X = self._generate_approximate_X_for_fiedler(12, 2, xp)
        lobpcg_w, lobpcg_V = sp.linalg.lobpcg(A, X, largest=False)
        return lobpcg_w, _eigen_vec_transform(lobpcg_V, xp)

    @testing.numpy_cupy_allclose(
        rtol=1e-5, atol=5e-3 if runtime.is_hip else 1e-3, sp_name='sp',
        contiguous_check=False)
    def test_random_initial_float32(self, xp, sp):
        """Check lobpcg in float32 for specific initial.
        """
        n = 50
        m = 4
        # Use dtype=float to avoid SciPy 1.17 FutureWarning for
        # integer input to ``diags``.
        vals = -xp.arange(1, n + 1, dtype=float)
        A = sp.diags([vals], [0], (n, n))
        A = A.astype(xp.float32)
        X = testing.shaped_random((n, m), xp=xp, seed=3)
        eigvals, vecs = sp.linalg.lobpcg(A, X, tol=1e-3, maxiter=50,
                                         verbosityLevel=1)
        return eigvals, _eigen_vec_transform(vecs, xp)

    @pytest.mark.xfail(
        runtime.is_hip and driver.get_build_version() >= 5_00_00000,
        reason='ROCm 5.0+ may have a bug')
    @pytest.mark.xfail(
        cupy.cuda.cusolver._getVersion() >= (11, 4, 5),  # CUDA 12.1.1+
        reason='cuSOLVER in CUDA 12.1+ may have a bug',
        strict=False,  # Seems only failing with Volta (V100 / T4)
    )
    def test_maxit_None(self):
        """Check lobpcg if maxit=None runs 20 iterations (the default)
        by checking the size of the iteration history output, which should
        be the number of iterations plus 2 (initial and final values).
        """
        def make(xp, sp):
            vals = -xp.arange(1, n + 1)
            A = sp.diags([vals], [0], (n, n))
            A = A.astype(xp.float32)
            X = testing.shaped_random((n, m), xp=xp, seed=1566950023)
            return A, X
        n = 50
        m = 4
        A, X = make(cupy, sparse)
        w, _, l_h = sparse.linalg.lobpcg(
            A, X, tol=1e-8, maxiter=None, retLambdaHistory=True)

        # Assert the eigenavlues against SciPy
        A_np, X_np = make(numpy, scipy.sparse)
        w_np, _ = scipy.sparse.linalg.lobpcg(
            A_np, X_np, tol=1e-8, maxiter=None)
        testing.assert_allclose(w, w_np, rtol=1e-5)

        # Assert the number of iterations
        assert len(l_h) == 22


@testing.with_requires('scipy')
@testing.parameterize(*testing.product({
    'A_sparsity': [True, False],
    'B_sparsity': [True, False],
    'A_dtype': [cupy.float32, cupy.float64],
    'preconditioner_sparsity': [True, False],
    'preconditioner_dtype': [None, cupy.float32, cupy.float64],
    'X_dtype': [cupy.float32, cupy.float64],
    'Y_dtype': [cupy.float32, cupy.float64],
    'sparse_format': ['coo', 'csr', 'csc']
}))
# test class for testing against diagonal matrices overall various data types
class TestLOBPCGForDiagInput:

    @pytest.fixture(autouse=True)
    def setUp(self):
        if runtime.is_hip:
            if driver.get_build_version() < 402:
                pytest.skip('syevj not available')
            if (((self.A_sparsity is True) or (self.B_sparsity is True)
                    or (self.preconditioner_sparsity is True))
                    and self.sparse_format == 'csc'):
                pytest.xfail('spMM not working')

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp',
                                 contiguous_check=False)
    @pytest.mark.slow
    def test_diagonal_data_types(self, xp, sp):
        """Check lobpcg for diagonal matrices for all matrix types.
        """
        n = 40
        m = 4
        # Define the generalized eigenvalue problem Av = cBv
        # where (c, v) is a generalized eigenpair,
        # and where we choose A  and B to be diagonal.
        vals = xp.arange(1, n + 1)
        # A and B matrices based on parametrization
        A = sp.diags([vals * vals], [0], (n, n), format=self.sparse_format)
        A = A.astype(xp.dtype(self.A_dtype))
        A = A if self.A_sparsity is True else A.toarray()

        B = sp.diags([vals], [0], (n, n), format=self.sparse_format)
        B = B if self.B_sparsity is True else B.toarray()

        M_LO = None
        if self.preconditioner_dtype is not None:
            M = sp.diags([1. / vals], [0], (n, n), format=self.sparse_format)
            M = M if self.preconditioner_sparsity else M.toarray()

            def fun(x):
                return M @ x
            # Define Preconditioner function as Linear Operator
            M_LO = sp.linalg.LinearOperator(matvec=fun,
                                            matmat=fun,
                                            shape=(n, n),
                                            dtype=xp.dtype(self.preconditioner_dtype))  # NOQA

        # Cannot be sparse array
        X = testing.shaped_random((n, m), xp=xp, dtype=xp.dtype(self.X_dtype),
                                  seed=1234)

        # Require that returned eigenvectors be in the orthogonal
        # complement of the first few standard basis vectors
        # (Cannot be sparse array)
        m_excluded = 3
        Y = xp.eye(n, m_excluded, dtype=xp.dtype(self.Y_dtype))
        # core call to lobpcg solver
        eigvals, eigvecs = sp.linalg.lobpcg(A, X, B=B, M=M_LO, Y=Y,
                                            tol=1e-4, maxiter=100,
                                            largest=False)
        return eigvals, _eigen_vec_transform(eigvecs, xp)


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc', 'coo'],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F']
}))
@testing.with_requires('scipy')
@pytest.mark.skipif(not cusparse.check_availability('csrsm2'),
                    reason='no working implementation')
class TestSplu:

    n = 10
    density = 0.5

    def _make_matrix(self, dtype, xp, sp, density=None):
        if density is None:
            density = self.density
        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, xp, dtype=dtype, scale=2 / self.n)
        mask = testing.shaped_random(a_shape, xp, dtype='f', scale=1)
        a[mask > density] = 0
        diag = xp.diag(xp.ones((self.n,), dtype=dtype))
        a = a + diag
        if self.format == 'csr':
            a = sp.csr_matrix(a)
        elif self.format == 'csc':
            a = sp.csc_matrix(a)
        elif self.format == 'coo':
            a = sp.coo_matrix(a)
        b_shape = (self.n,) if self.nrhs is None else (self.n, self.nrhs)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, order=self.order)
        return a, b

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_splu(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.splu(a).solve(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_factorized(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.factorized(a)(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_spilu(self, dtype, xp, sp):
        a, b = self._make_matrix(dtype, xp, sp)
        return sp.linalg.spilu(a).solve(b)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_spilu_0(self, dtype, xp, sp):
        # Note: We don't know how to compute ILU(0) with
        # scipy.sprase.linalg.spilu, so in this test we use a matrix where the
        # format is a sparse matrix but is actually a dense matrix.
        a, b = self._make_matrix(dtype, xp, sp, density=1.0)
        if xp == cupy:
            # Set fill_factor=1 to computes ILU(0) using cuSparse
            ainv = sp.linalg.spilu(a, fill_factor=1)
        else:
            ainv = sp.linalg.spilu(a)
        return ainv.solve(b)


@testing.parameterize(*testing.product({
    'damp': [0.0, 1.0, 2.0],
    'format': ['coo', 'csr', 'csc'],
    'm': [30, 40, 50],
    'n': [20, 30],
    'x0': [None, 'ones'],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestLsmr:

    density = 0.01

    def _make_matrix(self, xp, dtype):
        shape = (self.m, self.n)
        a = testing.shaped_random(shape, xp, dtype, scale=1)
        mask = testing.shaped_random(shape, xp, scale=1)
        a[mask > self.density] = 0
        return a

    def _make_normalized_vector(self, xp, dtype):
        b = testing.shaped_random((self.m,), xp, dtype, scale=1)
        return b / xp.linalg.norm(b)

    def _test_lsmr(self, xp, sp, a):
        b = self._make_normalized_vector(xp, a.dtype)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,))
        return sp.linalg.lsmr(a, b, x0=x0, damp=self.damp)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-1, atol=1e-1, sp_name='sp')
    def test_sparse(self, xp, sp, dtype):
        if runtime.is_hip and self.format in ('csr', 'csc'):
            pytest.xfail('may be buggy')  # trans=True

        if (self.damp == 0 and self.x0 == 'ones' and self.n != 20):
            pytest.skip()
        a = self._make_matrix(xp, dtype)
        a = sp.coo_matrix(a).asformat(self.format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_lsmr(xp, sp, a)[0]

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-1, atol=1e-1, sp_name='sp')
    def test_dense(self, xp, sp, dtype):
        if (self.damp == 0 and self.x0 == 'ones' and self.n != 20):
            pytest.skip()
        a = self._make_matrix(xp, dtype)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        return self._test_lsmr(xp, sp, a)[0]

    @testing.for_float_dtypes(no_float16=True)
    def test_invalid(self, dtype):
        if not (self.x0 is None and self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = self._make_matrix(xp, dtype)
            b = self._make_normalized_vector(xp, dtype)
            ng_a = xp.ones((self.m, ))
            with pytest.raises(ValueError):
                sp.linalg.lsmr(ng_a, b)
            ng_a = xp.ones((self.m, self.n, 1))
            with pytest.raises(ValueError):
                sp.linalg.lsmr(ng_a, b)
            ng_b = xp.ones((self.m + 1,))
            with pytest.raises(ValueError):
                sp.linalg.lsmr(a, ng_b)
            ng_b = xp.ones((self.m, 2))
            with pytest.raises(ValueError):
                sp.linalg.lsmr(a, ng_b)
            ng_x0 = xp.ones((self.n + 1,))
            with pytest.raises(ValueError):
                sp.linalg.lsmr(a, b, x0=ng_x0)


@testing.parameterize(*testing.product({
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'atol': [None, 'select-by-dtype'],
    'b_ndim': [1, 2],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestCgs:
    n = 30
    density = 0.33
    _atol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        dtype = numpy.dtype(dtype)
        shape = (self.n, 10)
        a = testing.shaped_random(shape, xp, dtype=dtype.char.lower(), scale=1)
        if dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(
                shape, xp, dtype=dtype.char.lower(), scale=1)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a @ a.conj().T
        a = a + xp.diag(xp.ones((self.n,), dtype=dtype.char.lower()))
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, dtype, xp):
        b = testing.shaped_random((self.n,), xp, dtype=dtype)
        return b / xp.linalg.norm(b)

    def _test_cgs(self, dtype, xp, sp, a, M):
        dtype = numpy.dtype(dtype)
        b = self._make_normalized_vector(dtype, xp)
        if self.b_ndim == 2:
            b = b.reshape(self.n, 1)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.n,), dtype=dtype)
        atol = 0.0
        if self.atol == 'select-by-dtype':
            atol = self._atol[dtype.char.lower()]
        return sp.linalg.cgs(a, b, x0=x0, M=M, atol=atol)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a, M = self._make_matrix(dtype, xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cgs(dtype, xp, sp, a, M)

    @pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, format, dtype, xp, sp):
        if runtime.is_hip and format == 'csc':
            pytest.xfail('may be buggy')  # trans=True

        a, M = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_cgs(dtype, xp, sp, a, M)

    @testing.with_requires('scipy')
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_empty(self, dtype, xp, sp):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        a = xp.empty((0, 0), dtype=dtype)
        b = xp.empty((0,), dtype=dtype)
        if self.atol is None and xp == numpy:
            return sp.linalg.cgs(a, b)
        else:
            return sp.linalg.cgs(a, b)

    @testing.for_dtypes('fdFD')
    def test_callback(self, dtype):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        xp, sp = cupy, sparse
        a, M = self._make_matrix(dtype, xp)
        b = self._make_normalized_vector(dtype, xp)
        is_called = False

        def callback(x):
            print(xp.linalg.norm(b - a @ x))
            nonlocal is_called
            is_called = True
        sp.linalg.cgs(a, b, callback=callback)
        assert is_called

    def test_invalid(self):
        if not (self.x0 is None and self.M is None and self.atol is None and
                self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix('f', xp)
            b = self._make_normalized_vector('f', xp)
            ng_a = xp.ones((self.n, ), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n + 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(ng_a, b, atol=self.atol)
            ng_b = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(a, ng_b, atol=self.atol)
            ng_b = xp.ones((self.n, 2), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(a, ng_b, atol=self.atol)
            ng_x0 = xp.ones((self.n + 1,), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.cgs(a, b, x0=ng_x0, atol=self.atol)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.cgs(a, b, M=ng_M, atol=self.atol)
        xp, sp = cupy, sparse
        b = self._make_normalized_vector('f', xp)
        ng_a = xp.ones((self.n, self.n), dtype='i')
        with pytest.raises(TypeError):
            sp.linalg.cgs(ng_a, b, atol=self.atol)


@testing.parameterize(*testing.product({
    'format': ['coo', 'csr', 'csc'],
    'm': [30, 40],
    'x0': [None, 'ones'],
    'M': [None, 'jacobi'],
    'shift': [0, 1],
    'use_linear_operator': [False, True],
}))
@testing.with_requires('scipy')
class TestMinres:

    density = 0.01

    def _make_matrix(self, xp):
        shape = (self.m, self.m)
        a = testing.shaped_random(shape, xp, scale=1)
        mask = testing.shaped_random(shape, xp, scale=1)
        a[mask > self.density] = 0
        M = None
        if self.M == 'jacobi':
            M = xp.diag(1.0 / xp.diag(a))
        return a, M

    def _make_normalized_vector(self, xp):
        b = testing.shaped_random((self.m,), xp, scale=1)
        return b / xp.linalg.norm(b)

    def _test_minres(self, xp, sp, a, M):
        b = self._make_normalized_vector(xp)
        x0 = None
        if self.x0 == 'ones':
            x0 = xp.ones((self.m,))
        return sp.linalg.minres(a, b, x0=x0, M=M)[0]

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, xp, sp):
        if runtime.is_hip and self.format == 'csc':
            pytest.xfail('may be buggy')  # trans=True
        a, M = self._make_matrix(xp)
        a = sp.coo_matrix(a).asformat(self.format)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
        if M is not None:
            M = sp.coo_matrix(M).asformat(self.format)
            if self.use_linear_operator:
                M = sp.linalg.aslinearoperator(M)
        return self._test_minres(xp, sp, a, M)

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, xp, sp):
        a, M = self._make_matrix(xp)
        if self.use_linear_operator:
            a = sp.linalg.aslinearoperator(a)
            if M is not None:
                M = sp.linalg.aslinearoperator(M)
        return self._test_minres(xp, sp, a, M)

    def test_invalid(self):
        if not (self.x0 is None and self.M is None
                and self.use_linear_operator is False):
            pytest.skip()
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a, M = self._make_matrix(xp)
            b = self._make_normalized_vector(xp)
            ng_a = xp.ones((self.m, ))
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_a = xp.ones((self.m, self.m + 1))
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_a = xp.ones((self.m, self.m, 1), dtype='f')
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_b = xp.ones((self.m + 1,))
            with pytest.raises(ValueError):
                sp.linalg.minres(a, ng_b)
            ng_b = xp.ones((self.m, 2))
            with pytest.raises(ValueError):
                sp.linalg.minres(a, ng_b)
            ng_x0 = xp.ones((self.m + 1,))
            with pytest.raises(ValueError):
                sp.linalg.minres(a, b, x0=ng_x0)
            ng_M = xp.diag(xp.ones((self.m + 1,)))
            with pytest.raises(ValueError):
                sp.linalg.minres(a, b, M=ng_M)

    def test_callback(self):
        if not (self.x0 is None and self.M is None and
                self.use_linear_operator is False):
            pytest.skip()
        xp, sp = cupy, sparse
        a, M = self._make_matrix(xp)
        b = self._make_normalized_vector(xp)
        is_called = False

        def callback(x):
            nonlocal is_called
            is_called = True
        sp.linalg.minres(a, b, callback=callback)
        assert is_called
