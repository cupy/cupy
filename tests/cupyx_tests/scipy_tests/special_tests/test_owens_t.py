from __future__ import annotations

import numpy

from cupy import testing


@testing.with_requires('scipy')
class TestOwensT:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_grid(self, xp, scp, dtype):
        h = xp.linspace(-10.0, 10.0, 41, dtype=dtype)
        a = xp.linspace(-10.0, 10.0, 41, dtype=dtype)
        h, a = xp.meshgrid(h, a)
        return scp.special.owens_t(h, a)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_wide_range(self, xp, scp, dtype):
        h = xp.logspace(-5, 2, 15, dtype=dtype)
        a = xp.logspace(-5, 5, 21, dtype=dtype)
        h, a = xp.meshgrid(h, a)
        return scp.special.owens_t(h, a)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_special_values(self, xp, scp, dtype):
        vals = [0.0, 1.0, -1.0, numpy.inf, -numpy.inf, numpy.nan]
        h, a = xp.meshgrid(xp.array(vals, dtype=dtype),
                           xp.array(vals, dtype=dtype))
        return scp.special.owens_t(h, a)
