from __future__ import annotations

import numpy

from cupy import testing


@testing.with_requires('scipy')
class TestTklmbda:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_grid(self, xp, scp, dtype):
        x = xp.linspace(-10.0, 10.0, 41, dtype=dtype)
        lmbda = xp.linspace(-2.0, 2.0, 41, dtype=dtype)
        x, lmbda = xp.meshgrid(x, lmbda)
        return scp.special.tklmbda(x, lmbda)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_special_values(self, xp, scp, dtype):
        vals = xp.array([0.0, 1.0, -1.0, numpy.inf, -numpy.inf, numpy.nan],
                        dtype=dtype)
        x, lmbda = xp.meshgrid(vals, vals)
        return scp.special.tklmbda(x, lmbda)
