from __future__ import annotations

from cupy import testing


@testing.with_requires('scipy')
class TestDawsn:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_real(self, xp, scp, dtype):
        x = xp.linspace(-100.0, 100.0, 201, dtype=dtype)
        return scp.special.dawsn(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_complex(self, xp, scp, dtype):
        x = xp.linspace(-5.0, 5.0, 21, dtype=dtype)
        y = xp.linspace(-5.0, 5.0, 21, dtype=dtype)
        x, y = xp.meshgrid(x, y)
        z = (x + 1j * y).ravel()
        return scp.special.dawsn(z)
