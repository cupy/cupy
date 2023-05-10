import numpy
import pytest

import cupy
from cupy import testing

try:
    import cupyx.scipy.special
except ImportError:
    pass


@testing.with_requires('scipy')
class TestEllipk:
    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-15)
    def test_basic_ellipk(self, xp, scp):
        x = xp.linspace(1e-14, 1, 101)
        return scp.special.ellipk(x)

    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-15)
    def test_basic_ellipkm1(self, xp, scp):
        x = xp.linspace(1e-14, 1, 101)
        return scp.special.ellipkm1(1./x)
