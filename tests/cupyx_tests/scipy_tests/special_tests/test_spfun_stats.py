from __future__ import annotations

import numpy

from cupy import testing


@testing.with_requires('scipy')
class TestPoissonBinom:
    rng = numpy.random.default_rng(1234)
    p = rng.uniform(0, 1, (1000, 10))
    k = rng.integers(-10, 10, size=(20, 1000))

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(
        scipy_name='scp',
        rtol={numpy.float32: 1e-6, 'default': 1e-14},
        atol={numpy.float32: 1e-6, 'default': 1e-14}
    )
    def test_poisson_binom_pmf(self, xp, scp, dtype):
        import scipy.special  # NOQA
        p = xp.asarray(self.p, dtype=dtype)
        k = xp.asarray(self.k)
        return scp.special._spfun_stats._poisson_binom_pmf(k, p)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(
        scipy_name='scp',
        rtol={numpy.float32: 1e-6, 'default': 1e-14},
        atol={numpy.float32: 1e-6, 'default': 1e-14}
    )
    def test_poisson_binom_cdf(self, xp, scp, dtype):
        import scipy.special  # NOQA
        p = xp.asarray(self.p, dtype=dtype)
        k = xp.asarray(self.k)
        return scp.special._spfun_stats._poisson_binom_cdf(k, p)
