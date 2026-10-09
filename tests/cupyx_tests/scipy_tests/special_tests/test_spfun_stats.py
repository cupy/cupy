from __future__ import annotations

import numpy
import cupy

from cupy import testing
from cupyx.scipy.special import poisson_binom_cdf


class TestPoissonBinomCDF:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(
        scipy_name='scp',
        rtol={numpy.float32: 1e-6, 'default': 1e-14},
        atol={numpy.float32: 1e-6, 'default': 1e-14}
    )
    @testing.with_requires('scipy>=2.0.0')
    def test_against_scipy(self, xp, scp, dtype):
        import scipy.special  # NOQA

        rng = numpy.random.default_rng(1234)
        p = xp.asarray(rng.uniform(0, 1, (1000, 10)), dtype=dtype)
        k = xp.asarray(rng.integers(-10, 10, size=(20, 1000)))
        return scp.special.poisson_binom_cdf(k, p)

    @testing.for_dtypes('fd')
    def test_values(self, dtype):
        p = cupy.asarray([0.2, 0.4, 0.6], dtype=dtype)
        k = cupy.asarray([0, 1, 2, 3])
        actual = poisson_binom_cdf(k, p)
        # Hardcoded reference taken from SciPy 2.0.0dev so that test
        # coverage can exist prior to release of SciPy 2.0.0.
        desired = cupy.asarray(
            [0.192, 0.656, 0.9520000000000001, 1.0], dtype=dtype)
        testing.assert_allclose(actual, desired)
        assert actual.dtype == desired.dtype
