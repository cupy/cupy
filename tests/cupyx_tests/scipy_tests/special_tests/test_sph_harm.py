import cupy as cp
import pytest

import cupyx.scipy.special  # NOQA
from cupy import testing
from cupy.testing import numpy_cupy_allclose


def _get_harmonic_list(degree_max):
    """Generate list of all spherical harmonics up to degree_max."""
    harmonic_list = []
    for degree in range(degree_max + 1):
        for order in range(-degree, degree + 1):
            harmonic_list.append((order, degree))
    return harmonic_list


@testing.with_requires("scipy")
class TestBasic():

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.parametrize("m, n", _get_harmonic_list(degree_max=5))
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-7, atol=1e-10)
    def test_sph_harm(self, xp, scp, dtype, m, n):
        theta = xp.linspace(0, 2 * cp.pi)
        phi = xp.linspace(0, cp.pi)
        theta, phi = xp.meshgrid(theta, phi)
        return scp.special.sph_harm(m, n, theta, phi)

    @testing.with_requires("scipy>=1.15.0")
    @pytest.mark.parametrize("m, n", _get_harmonic_list(degree_max=5))
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp", rtol=1e-7, atol=1e-10)
    def test_sph_harm_y(self, xp, scp, dtype, m, n):
        theta = xp.linspace(0, cp.pi)
        phi = xp.linspace(0, 2 * cp.pi)
        theta, phi = xp.meshgrid(theta, phi)
        return scp.special.sph_harm_y(n, m, theta, phi)
