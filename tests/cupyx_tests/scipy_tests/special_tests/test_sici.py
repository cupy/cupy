from __future__ import annotations
from cupy import testing

try:
    import cupyx.scipy.special    # NOQA
except ImportError:
    pass


@testing.with_requires('scipy')
class TestSici:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_real(self, xp, scp, dtype):
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        return scp.special.sici(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_complex(self, xp, scp, dtype):
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        y = xp.linspace(-20, 20, 12, dtype=dtype)
        x, y = xp.meshgrid(x, y)
        z = (x + y*1j).ravel()
        return scp.special.sici(z)


@testing.with_requires('scipy')
class TestShichi:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_real(self, xp, scp, dtype):
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        return scp.special.shichi(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_complex(self, xp, scp, dtype):
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        y = xp.linspace(-20, 20, 12, dtype=dtype)
        x, y = xp.meshgrid(x, y)
        z = (x + y*1j).ravel()
        return scp.special.shichi(z)
