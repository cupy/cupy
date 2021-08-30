import unittest

import cupy
from cupy import testing
import cupyx.scipy.linalg

import numpy

try:
    import scipy.linalg

    _scipy_available = True
except ImportError:
    _scipy_available = False


@testing.gpu
@testing.parameterize(
    *testing.product(
        {
            "shape": [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10)],
        }
    )
)
@testing.fix_random()
@testing.with_requires("scipy")
class TestLUFactor(unittest.TestCase):
    @testing.for_dtypes("fdFD")
    def test_lu_factor(self, dtype):
        if self.shape[0] != self.shape[1]:
            self.skipTest("skip non-square tests since scipy.expm requires square")
        a_cpu = testing.shaped_random(self.shape, numpy, dtype=dtype)
        a_gpu = cupy.asarray(a_cpu)
        result_cpu = scipy.linalg.expm(a_cpu)
        result_gpu = cupyx.scipy.linalg.expm(a_gpu)
        assert len(result_cpu) == len(result_gpu)
        assert result_cpu[0].dtype == result_gpu[0].dtype
        assert result_cpu[1].dtype == result_gpu[1].dtype
        cupy.testing.assert_allclose(result_cpu[0], result_gpu[0], atol=1e-5)
        cupy.testing.assert_array_equal(result_cpu[1], result_gpu[1])
