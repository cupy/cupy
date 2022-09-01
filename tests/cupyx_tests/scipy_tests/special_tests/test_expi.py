import cupyx.scipy.special  # NOQA

from cupy import testing
from cupy.testing import numpy_cupy_allclose


@testing.with_requires("scipy")
class TestExpi:

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_zero(self, xp, scp):
        return scp.special.expi(0.0)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_negative(self, xp, scp):
        return scp.special.expi(-4.2)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_large_negative(self, xp, scp):
        return scp.special.expi(-1000)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_positive(self, xp, scp):
        return scp.special.expi(4.2)

    @numpy_cupy_allclose(scipy_name="scp")
    def test_expi_large_positive(self, xp, scp):
        return scp.special.expi(1000)

    @testing.for_dtypes("f")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-5)
    def test_expi_linspace_float32(self, xp, scp, dtype):
        x = xp.linspace(-10, 60, 1000, dtype=dtype)
        return scp.special.expi(x)

    @testing.for_dtypes("d")
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-12)
    def test_expi_linspace_float64(self, xp, scp, dtype):
        x = xp.linspace(-10, 100, 1000, dtype=dtype)
        return scp.special.expi(x)
