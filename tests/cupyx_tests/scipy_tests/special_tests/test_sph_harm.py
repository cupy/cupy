import cupy as cp
from cupy.testing import assert_allclose
import cupyx.scipy.special as sc


# TODO: update/expand sph_harm tests. The below is adapted from SciPy


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
            sc.sph_harm(m, n, theta, phi),
            harm(theta, phi),
            rtol=1e-15,
            atol=1e-15,
            err_msg="Y^{}_{} incorrect".format(m, n),
        )
