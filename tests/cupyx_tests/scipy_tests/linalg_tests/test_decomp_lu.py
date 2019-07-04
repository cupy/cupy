import unittest

import numpy
import cupy
from cupy import cuda
from cupy import testing
import cupyx.scipy.linalg
if cupyx.scipy._scipy_available:
    import scipy.linalg


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(1, 1), (2, 2), (3, 3), (5, 5), (1, 5), (5, 1), (2, 5), (5, 2)],
}))
@testing.fix_random()
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.with_requires('scipy')
class TestLUFactor(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def test_lu_factor(self, dtype):
        if self.shape[0] != self.shape[1]:
            # skip non-square tests since scipy.lu_factor requires square
            return unittest.SkipTest()
        array = numpy.random.randn(*self.shape)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = scipy.linalg.lu_factor(a_cpu)
        result_gpu = cupyx.scipy.linalg.lu_factor(a_gpu)
        self.assertEqual(len(result_cpu), len(result_gpu))
        self.assertEqual(result_cpu[0].dtype, result_gpu[0].dtype)
        self.assertEqual(result_cpu[1].dtype, result_gpu[1].dtype)
        cupy.testing.assert_allclose(result_cpu[0], result_gpu[0], atol=1e-5)
        cupy.testing.assert_array_equal(result_cpu[1], result_gpu[1])

    @testing.for_float_dtypes(no_float16=True)
    def test_lu_factor_reconstruction(self, dtype):
        m, n = self.shape
        array = cupy.random.randn(m, n, dtype=dtype)
        lu, piv = cupyx.scipy.linalg.lu_factor(array)
        # extract ``L`` and ``U`` from ``lu``
        L = cupy.tril(lu, k=-1)
        cupy.fill_diagonal(L, 1.)
        if m < n:
            L = L[:, :m]
        U = cupy.triu(lu)
        if m > n:
            U = U[:n, :]
        # apply pivot (on CPU since slaswp is not available in cupy)
        piv = cupy.asnumpy(piv)
        rows = numpy.arange(array.shape[0])
        for i, row in enumerate(piv):
            if i != row:
                rows[i], rows[row] = rows[row], rows[i]
        # revert pivot
        reversed_piv = numpy.empty_like(rows)
        reversed_piv[rows] = numpy.arange(rows.size)
        # swap L
        L = L[reversed_piv]
        # check that reconstruction is close to original
        reconstructed = L.dot(U)
        cupy.testing.assert_allclose(reconstructed, array, atol=1e-5)


@testing.gpu
@testing.parameterize(*testing.product({
    'trans': [0, 1, 2],
    'shapes': [((4, 4), (4,)), ((5, 5), (5, 2))],
}))
@testing.fix_random()
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.with_requires('scipy')
class TestLUSolve(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_lu_solve(self, xp, scp, dtype):
        a_shape, b_shape = self.shapes
        A = testing.shaped_random(a_shape, xp, dtype=dtype)
        b = testing.shaped_random(b_shape, xp, dtype=dtype)
        lu = scp.linalg.lu_factor(A)
        return scp.linalg.lu_solve(lu, b, trans=self.trans)
