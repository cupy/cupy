import cupyx.scipy.special  # NOQA

from cupy import testing
from cupy.testing import numpy_cupy_allclose


@testing.with_requires("scipy")
class TestExpn:

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn(self, xp, scp, dtype):
        return scp.special.expn(-1, 1.0)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_2(self, xp, scp, dtype):
        return scp.special.expn(1, -1.0)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_large_values(self, xp, scp, dtype):
        return scp.special.expn(500, 10)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_large_values_2(self, xp, scp, dtype):
        return scp.special.expn(10, 500)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_zero_values(self, xp, scp, dtype):
        return scp.special.expn(1.0, 0)

    @testing.for_dtypes("efdFD")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_zero_values_2(self, xp, scp, dtype):
        return scp.special.expn(0.0, 2)

    @testing.for_dtypes("edf")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_array_inputs(self, xp, scp, dtype):
        x = testing.shaped_arange((5, 4, 2), xp, dtype)
        n = testing.shaped_arange((5, 1, 2), xp, dtype)
        return scp.special.expn(n, x)

    @testing.for_dtypes("edf")
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_array_inputs_2(self, xp, scp, dtype):
        x = testing.shaped_random((5, 3), xp, dtype)
        n = testing.shaped_random((5, 3), xp, dtype)
        return scp.special.expn(n, x)
