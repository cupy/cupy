from __future__ import annotations

from cupy import testing


@testing.with_requires('scipy')
class TestVoigtProfile:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_values(self, xp, scp, dtype):
        x = xp.linspace(-100.0, 100.0, 201, dtype=dtype)[:, None, None]
        sigma = xp.asarray([0.0, 0.1, 1.0, 10.0], dtype=dtype)[None, :, None]
        gamma = xp.asarray([0.0, 0.1, 1.0, 10.0], dtype=dtype)[None, None, :]
        return scp.special.voigt_profile(x, sigma, gamma)
