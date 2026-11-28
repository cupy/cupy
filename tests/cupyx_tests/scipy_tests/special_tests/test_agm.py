from __future__ import annotations

import numpy

from cupy import testing


@testing.with_requires('scipy')
class TestAgm:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_positive(self, xp, scp, dtype):
        a = xp.logspace(-5, 5, 21, dtype=dtype)
        a, b = xp.meshgrid(a, a)
        return scp.special.agm(a, b)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_special_values(self, xp, scp, dtype):
        vals = [0.0, -0.0, 1.0, -1.0, numpy.inf, -numpy.inf, numpy.nan]
        a, b = xp.meshgrid(xp.array(vals, dtype=dtype),
                           xp.array(vals, dtype=dtype))
        return scp.special.agm(a, b)
