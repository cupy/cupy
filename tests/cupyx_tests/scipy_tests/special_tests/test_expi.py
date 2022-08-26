import cupyx.scipy.special  # NOQA

from cupy import testing
from cupy.testing import numpy_cupy_allclose


@testing.gpu
@testing.with_requires("scipy")
class TestExpi:

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_zero(self, xp, scp):
        return scp.special.expi(0.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_negative(self, xp, scp):
        return scp.special.expi(-42.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_large_negative(self, xp, scp):
        return scp.special.expi(-1000)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_positive(self, xp, scp):
        return scp.special.expi(42.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_large_positive(self, xp, scp):
        return scp.special.expi(1000)

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_array_inputs(self, xp, scp, dtype):
        x = testing.shaped_arange((8, 9, 10), xp, dtype)
        return scp.special.expi(x)

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-15)
    def test_expi_linspace(self, xp, scp, dtype):
        x = xp.linspace(-1000, 1000, 500, dtype=dtype)
        return scp.special.expi(x)

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp", atol=1e-9)
    def test_expi_random(self, xp, scp, dtype):
        x = testing.shaped_random((5, 5, 5), xp, dtype=dtype)
        return scp.special.expi(x)
