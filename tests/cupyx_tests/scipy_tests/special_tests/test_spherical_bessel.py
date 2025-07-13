from __future__ import annotations

import numpy

from cupy import testing
import cupyx.scipy.special  # NOQA

try:
    import scipy.special  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy>=1.15")
class TestSphericalBessel:

    @testing.for_dtypes('il', name='order_dtype')
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_spherical_yn_inf_and_nan(self, xp, scp, dtype, order_dtype):
        n = xp.arange(0, 100, dtype=order_dtype)
        a = xp.array([numpy.nan, numpy.inf, -numpy.inf], dtype=dtype)
        return scp.special.spherical_yn(n[xp.newaxis, :], a[:, xp.newaxis])

    @testing.for_dtypes('il', name='order_dtype')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_spherical_yn_linspace(self, xp, scp, dtype, order_dtype):
        n = testing.shaped_arange((10,), xp=xp, dtype=order_dtype)
        a = testing.shaped_linspace(-10, 10, 100, xp=xp, dtype=dtype)
        return scp.special.spherical_yn(n[xp.newaxis, :], a[:, xp.newaxis])
