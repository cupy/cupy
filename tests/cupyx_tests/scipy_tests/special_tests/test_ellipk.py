from cupy import testing

try:
    import cupyx.scipy.special    # NOQA
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


@testing.with_requires('scipy')
class TestEllipj:
    @testing.numpy_cupy_allclose(scipy_name="scp", rtol=1e-13)
    def test_basic(self, xp, scp):
        el = scp.special.ellipj(0.2, 0)
        return el


@testing.with_requires('scipy')
class TestEllipkinc:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15)
    def test_values(self, xp, scp, dtype):
        phi = xp.linspace(-xp.pi, xp.pi, 5)
        m = xp.linspace(-1.0, 1.0, 5)
        phi, m = xp.meshgrid(phi, m)
        phi, m = phi.ravel(), m.ravel()

        return scp.special.ellipkinc(phi, m)


@testing.with_requires('scipy')
class TestEllipeinc:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-15)
    def test_values(self, xp, scp, dtype):
        phi = xp.linspace(-xp.pi, xp.pi, 5)
        m = xp.linspace(-1.0, 1.0, 5)
        phi, m = xp.meshgrid(phi, m)
        phi, m = phi.ravel(), m.ravel()

        return scp.special.ellipeinc(phi, m)
