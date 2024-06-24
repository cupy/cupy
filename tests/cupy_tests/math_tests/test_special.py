import unittest
import pytest

import cupy
from cupy import testing


class TestSpecial(unittest.TestCase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(rtol=1e-3)
    def test_i0(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.i0(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-3)
    def test_sinc(self, xp, dtype):

        if dtype in [cupy.float16, cupy.float32, cupy.complex64]:
            pytest.xfail(reason="XXX: np2.0: numpy 1.26 uses a wrong "
                                "promotion; numpy 2.0 is OK")

        a = testing.shaped_random((2, 3), xp, dtype, scale=1)
        return xp.sinc(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    def test_sinc_zero(self, dtype):
        a = cupy.sinc(cupy.zeros(1, dtype=dtype))
        testing.assert_array_equal(a, cupy.ones(1, dtype=dtype))
