import unittest

import cupy
from cupy import testing
import cupyx.scipy.linalg

import numpy
import pytest

try:
    import scipy.linalg
    _scipy_available = True
except ImportError:
    _scipy_available = False


@testing.parameterize(*testing.product({
    'trans': [0, 1, 2, 'N', 'T', 'C'],
    'lower': [True, False],
    'unit_diagonal': [True, False],
    'overwrite_b': [True, False],
    'check_finite': [True, False],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSolveTriangular(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    def check_x(self, a_shape, b_shape, dtype):
        a_cpu = numpy.random.randint(1, 10, size=a_shape).astype(dtype)
        b_cpu = numpy.random.randint(1, 10, size=b_shape).astype(dtype)
        a_cpu = numpy.tril(a_cpu)

        if self.lower is False:
            a_cpu = a_cpu.T
        if self.unit_diagonal is True:
            numpy.fill_diagonal(a_cpu, 1)

        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        a_gpu_copy = a_gpu.copy()
        b_gpu_copy = b_gpu.copy()
        result_cpu = scipy.linalg.solve_triangular(
            a_cpu, b_cpu, trans=self.trans, lower=self.lower,
            unit_diagonal=self.unit_diagonal, overwrite_b=self.overwrite_b,
            check_finite=self.check_finite)
        result_gpu = cupyx.scipy.linalg.solve_triangular(
            a_gpu, b_gpu, trans=self.trans, lower=self.lower,
            unit_diagonal=self.unit_diagonal, overwrite_b=self.overwrite_b,
            check_finite=self.check_finite)
        assert result_cpu.dtype == result_gpu.dtype
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        cupy.testing.assert_array_equal(a_gpu_copy, a_gpu)
        if not self.overwrite_b:
            cupy.testing.assert_array_equal(b_gpu_copy, b_gpu)

    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))
        self.check_x((5, 5), (5, 5))

    def check_shape(self, a_shape, b_shape):
        for xp, sp in ((numpy, scipy), (cupy, cupyx.scipy)):
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            with pytest.raises(ValueError):
                sp.linalg.solve_triangular(
                    a, b, trans=self.trans, lower=self.lower,
                    unit_diagonal=self.unit_diagonal,
                    overwrite_b=self.overwrite_b,
                    check_finite=self.check_finite)

    def test_invalid_shape(self):
        self.check_shape((2, 3), (4,))
        self.check_shape((3, 3), (2,))
        self.check_shape((3, 3), (2, 2))
        self.check_shape((3, 3, 4), (3,))

    def check_infinite(self, a_shape, b_shape):
        for xp, sp in ((numpy, scipy), (cupy, cupyx.scipy)):
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            a[(0,) * a.ndim] = numpy.inf
            b[(0,) * b.ndim] = numpy.inf
            with pytest.raises(ValueError):
                sp.linalg.solve_triangular(
                    a, b, trans=self.trans, lower=self.lower,
                    unit_diagonal=self.unit_diagonal,
                    overwrite_b=self.overwrite_b,
                    check_finite=self.check_finite)

    def test_infinite(self):
        if self.check_finite:
            self.check_infinite((4, 4), (4,))
            self.check_infinite((5, 5), (5, 2))
            self.check_infinite((5, 5), (5, 5))
