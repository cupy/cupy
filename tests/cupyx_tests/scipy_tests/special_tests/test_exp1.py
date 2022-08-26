import cupyx.scipy.special  # NOQA

from cupy import testing
from cupy.testing import numpy_cupy_allclose


@testing.gpu
@testing.with_requires("scipy")
class TestExp1:

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_zero(self, xp, scp):
        return scp.special.exp1(0.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_negative(self, xp, scp):
        return scp.special.exp1(-42.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_large_negative(self, xp, scp):
        return scp.special.exp1(-1000)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_positive(self, xp, scp):
        return scp.special.exp1(42.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_large_positive(self, xp, scp):
        return scp.special.exp1(1000)

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expn_array_inputs(self, xp, scp, dtype):
        x = testing.shaped_arange((8, 9, 10), xp, dtype)
        return scp.special.exp1(x)

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-15)
    def test_expn_array_inputs(self, xp, scp, dtype):
        x = xp.linspace(-1000, 1000, 1000, dtype=dtype)
        return scp.special.exp1(x)
