import unittest
import warnings

import numpy
import cupy
from cupy import testing
import cupyx.scipy.linalg
if cupyx.scipy._scipy_available:
    import scipy.linalg


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 2), (3, 3), (5, 5), (1, 5), (5, 1), (2, 5), (5, 2)],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLUFactor(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    def test_lu_factor(self, dtype):
        if self.shape[0] != self.shape[1]:
            # skip non-square tests since scipy.lu_factor requires square
            return unittest.SkipTest()
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
            return unittest.SkipTest()
        A = testing.shaped_random(self.shape, cupy, dtype=dtype)
        A -= A.mean(axis=0, keepdims=True)
        A -= A.mean(axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.check_lu_factor_reconstruction(A)


@testing.gpu
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


@testing.gpu
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
