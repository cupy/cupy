from __future__ import annotations

import numpy

from cupy import testing


@testing.with_requires('scipy')
class TestNdtriExp:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_wide_range(self, xp, scp, dtype):
        # Spans all three branches: y < -2, -2 <= y <= log1p(-exp(-2)),
        # and y close to 0.
        y = -xp.logspace(-30, 30, 121, dtype=dtype)
        return scp.special.ndtri_exp(y)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-6, scipy_name='scp')
    def test_special_values(self, xp, scp, dtype):
        y = xp.array([0.0, -0.0, 1.0, -numpy.inf, numpy.inf, numpy.nan],
                     dtype=dtype)
        return scp.special.ndtri_exp(y)
