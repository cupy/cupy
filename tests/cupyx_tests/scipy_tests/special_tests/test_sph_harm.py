import cupy as cp
import pytest
import scipy.special  # NOQA

import cupyx.scipy.special
from cupy import testing
from cupy.testing import assert_allclose, numpy_cupy_allclose


def test_first_harmonics():
    # Test against explicit representations of the first four
    # spherical harmonics which use `theta` as the azimuthal angle,
    # `phi` as the polar angle, and include the Condon-Shortley
    # phase.

    # Notation is Ymn
    def Y00(theta, phi):
        return 0.5 * cp.sqrt(1 / cp.pi)

    def Yn11(theta, phi):
        return (
            0.5 * cp.sqrt(3 / (2 * cp.pi)) * cp.exp(-1j * theta) * cp.sin(phi)
        )

    def Y01(theta, phi):
        return 0.5 * cp.sqrt(3 / cp.pi) * cp.cos(phi)

    def Y11(theta, phi):
        return (
            -0.5 * cp.sqrt(3 / (2 * cp.pi)) * cp.exp(1j * theta) * cp.sin(phi)
        )

    harms = [Y00, Yn11, Y01, Y11]
    m = [0, -1, 0, 1]
    n = [0, 1, 1, 1]

    theta = cp.linspace(0, 2 * cp.pi)
    phi = cp.linspace(0, cp.pi)
    theta, phi = cp.meshgrid(theta, phi)

    for harm, m, n in zip(harms, m, n):
        assert_allclose(
            cupyx.scipy.special.sph_harm(m, n, theta, phi),
            harm(theta, phi),
            rtol=1e-15,
            atol=1e-15,
            err_msg="Y^{}_{} incorrect".format(m, n),
        )


def _get_harmonic_list(degree_max):
    """Generate list of all spherical harmonics up to degree_max."""
    harmonic_list = []
    for degree in range(degree_max + 1):
        for order in range(-degree, degree + 1):
            harmonic_list.append((order, degree))
    return harmonic_list


@testing.gpu
@testing.with_requires("scipy")
class TestBasic():

    @pytest.mark.parametrize("m, n", _get_harmonic_list(degree_max=5))
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupy_allclose(scipy_name="scp")
    def test_sph_harm_higher_order(self, xp, scp, dtype, m, n):
        theta = xp.linspace(0, 2 * cp.pi)
        phi = xp.linspace(0, cp.pi)
        theta, phi = xp.meshgrid(theta, phi)
        return scp.special.sph_harm(m, n, theta, phi)
