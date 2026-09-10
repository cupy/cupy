from __future__ import annotations

import unittest

from cupy import testing
import cupyx.scipy.special  # NOQA
import pytest

try:
    import scipy.special  # NOQA
except ImportError:
    pass


# NOTE: float16 is left out of the dtype sweeps below.  SciPy 1.18 resolves
# float16 input to the float32 loop of the `scipy.special` ufuncs and returns
# float32, whereas CuPy's ufuncs declare `e->d` and return float64.
# TODO: switch those loops to `e->f` and restore float16 coverage before the
# minimum SciPy version is 1.18.
@testing.with_requires('scipy>=1.15')
class TestZetac(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):

        a = testing.shaped_arange((2, 3), xp, dtype)

        return scp.special.zetac(a)

    @testing.for_all_dtypes(no_complex=True, no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):

        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        a = xp.linspace(-30, 30, 1000, dtype=dtype)

        return scp.special.zetac(a)

    @testing.for_all_dtypes(no_complex=True, no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, scipy_name='scp')
    def test_linspace_small_negative_x(self, xp, scp, dtype):

        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        a = xp.linspace(-0.01, 0, 1000, dtype=dtype)

        return scp.special.zetac(a)

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):

        return scp.special.zetac(dtype(3.5))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        if xp.dtype(dtype).kind in 'iu':
            pytest.skip()
        x = xp.array([-xp.inf, xp.nan, xp.inf]).astype(dtype)
        a = xp.tile(x, (3, 3))

        return scp.special.zetac(a)
