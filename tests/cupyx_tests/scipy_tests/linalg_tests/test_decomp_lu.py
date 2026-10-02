from __future__ import annotations

import unittest
import warnings

import numpy
import cupy
from cupy import testing
import cupyx.scipy.linalg
if cupyx.scipy._scipy_available:
    import scipy.linalg


# TODO: After the feature is released
# requires_scipy_linalg_backend = testing.with_requires('scipy>=1.x.x')
requires_scipy_linalg_backend = unittest.skip(
    'scipy.linalg backend feature has not been released'
)


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 2), (3, 3), (5, 5), (1, 5), (5, 1), (2, 5), (5, 2)],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLUFactor(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    def test_lu_factor(self, dtype):
        if self.shape[0] != self.shape[1]:
            self.skipTest(
                'skip non-square tests since scipy.lu_factor requires square')
        a_cpu = testing.shaped_random(self.shape, numpy, dtype=dtype)
        a_gpu = cupy.asarray(a_cpu)
        result_cpu = scipy.linalg.lu_factor(a_cpu)
        result_gpu = cupyx.scipy.linalg.lu_factor(a_gpu)
        assert len(result_cpu) == len(result_gpu)
        assert result_cpu[0].dtype == result_gpu[0].dtype
        assert result_cpu[1].dtype == result_gpu[1].dtype
        cupy.testing.assert_allclose(result_cpu[0], result_gpu[0], atol=1e-5)
        cupy.testing.assert_array_equal(result_cpu[1], result_gpu[1])

    def check_lu_factor_reconstruction(self, A):
        m, n = self.shape
        lu, piv = cupyx.scipy.linalg.lu_factor(A)
        # extract ``L`` and ``U`` from ``lu``
        L = cupy.tril(lu, k=-1)
        cupy.fill_diagonal(L, 1.)
        L = L[:, :m]
        U = cupy.triu(lu)
        U = U[:n, :]
        # check output shapes
        assert lu.shape == (m, n)
        assert L.shape == (m, min(m, n))
        assert U.shape == (min(m, n), n)
        assert piv.shape == (min(m, n),)
        # apply pivot (on CPU since slaswp is not available in cupy)
        piv = cupy.asnumpy(piv)
        rows = numpy.arange(m)
        for i, row in enumerate(piv):
            if i != row:
                rows[i], rows[row] = rows[row], rows[i]
        PA = A[rows]
        # check that reconstruction is close to original
        LU = L.dot(U)
        cupy.testing.assert_allclose(LU, PA, atol=1e-5)

    @testing.for_dtypes('fdFD')
    def test_lu_factor_reconstruction(self, dtype):
        A = testing.shaped_random(self.shape, cupy, dtype=dtype)
        self.check_lu_factor_reconstruction(A)

    @testing.for_dtypes('fdFD')
    def test_lu_factor_reconstruction_singular(self, dtype):
        if self.shape[0] != self.shape[1]:
            self.skipTest(
                'skip non-square tests since scipy.lu_factor requires square')
        A = testing.shaped_random(self.shape, cupy, dtype=dtype)
        A -= A.mean(axis=0, keepdims=True)
        A -= A.mean(axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.check_lu_factor_reconstruction(A)


@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 2), (3, 3), (5, 5), (1, 5), (5, 1), (2, 5), (5, 2)],
    'permute_l': [False, True],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLU(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    def test_lu(self, dtype):
        a_cpu = testing.shaped_random(self.shape, numpy, dtype=dtype)
        a_gpu = cupy.asarray(a_cpu)
        result_cpu = scipy.linalg.lu(a_cpu, permute_l=self.permute_l)
        result_gpu = cupyx.scipy.linalg.lu(a_gpu, permute_l=self.permute_l)
        assert len(result_cpu) == len(result_gpu)
        if not self.permute_l:
            # check permutation matrix
            result_cpu = list(result_cpu)
            result_gpu = list(result_gpu)
            P_cpu = result_cpu.pop(0)
            P_gpu = result_gpu.pop(0)
            cupy.testing.assert_array_equal(P_gpu, P_cpu)
        cupy.testing.assert_allclose(result_gpu[0], result_cpu[0], atol=1e-5)
        cupy.testing.assert_allclose(result_gpu[1], result_cpu[1], atol=1e-5)

    @testing.for_dtypes('fdFD')
    def test_lu_reconstruction(self, dtype):
        m, n = self.shape
        A = testing.shaped_random(self.shape, cupy, dtype=dtype)
        if self.permute_l:
            PL, U = cupyx.scipy.linalg.lu(A, permute_l=self.permute_l)
            PLU = PL @ U
        else:
            P, L, U = cupyx.scipy.linalg.lu(A, permute_l=self.permute_l)
            PLU = P @ L @ U
        # check that reconstruction is close to original
        cupy.testing.assert_allclose(PLU, A, atol=1e-5)


@testing.parameterize(*testing.product({
    'trans': [0, 1, 2],
    'shapes': [((4, 4), (4,)), ((5, 5), (5, 2))],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLUSolve(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_lu_solve(self, xp, scp, dtype):
        a_shape, b_shape = self.shapes
        A = testing.shaped_random(a_shape, xp, dtype=dtype)
        b = testing.shaped_random(b_shape, xp, dtype=dtype)
        lu = scp.linalg.lu_factor(A)
        return scp.linalg.lu_solve(lu, b, trans=self.trans)

    @requires_scipy_linalg_backend
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_lu_solve_backend(self, xp, dtype):
        a_shape, b_shape = self.shapes
        A = testing.shaped_random(a_shape, xp, dtype=dtype)
        b = testing.shaped_random(b_shape, xp, dtype=dtype)
        if xp is numpy:
            lu = scipy.linalg.lu_factor(A)
            backend = 'scipy'
        else:
            lu = cupyx.scipy.linalg.lu_factor(A)
            backend = cupyx.scipy.linalg
        with scipy.linalg.set_backend(backend):
            out = scipy.linalg.lu_solve(lu, b, trans=self.trans)
        return out


class TestLUIndexOverflow(unittest.TestCase):

    @testing.slow
    def test_laswp_large_offset(self):
        # _cupy_split_lu and _cupy_laswp share a get_index() device helper
        # that used to compute flat offsets in 32-bit int, overflowing for
        # any matrix with m * n > INT32_MAX (2,147,483,647). 46341 x 46341
        # is the smallest square shape that crosses that threshold
        # (46341**2 == 2,147,488,281). Using uint8 keeps the array at ~2GB
        # instead of the 16GB+ a float64 matrix of this shape would need.
        #
        # Only _cupy_laswp is exercised directly here: _cupy_split_lu shares
        # the same helper but needs three large buffers (LU, L, U) to run,
        # which doesn't fit in this machine's VRAM budget.
        from cupyx.scipy.linalg._decomp_lu import _cupy_laswp

        m = n = 46341
        assert m * n > 2**31 - 1
        A = cupy.zeros((m, n), dtype=cupy.uint8)
        A[0, :] = 1
        A[m - 1, :] = 2

        ipiv = cupy.arange(m, dtype=cupy.int32)
        ipiv[m - 1] = 0

        _cupy_laswp(A, m - 1, m - 1, ipiv, 1)

        # The row-major flat offset for (row=m-1, col) crosses INT32_MAX
        # partway through the row (around col == 41708), so checking the
        # full row covers both the in-range and previously-overflowing
        # columns.
        assert bool(cupy.all(A[0, :] == 2))
        assert bool(cupy.all(A[m - 1, :] == 1))
