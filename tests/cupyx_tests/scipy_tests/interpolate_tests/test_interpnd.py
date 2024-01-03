
from cupy import testing
import cupyx.scipy.interpolate  # NOQA
import cupyx.scipy.spatial  # NOQA

try:
    import scipy.interpolate  # NOQA
except ImportError:
    pass

try:
    import scipy.spatial  # NOQA
except ImportError:
    pass


class TestLinearNDInterpolator:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoketest(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)

        yi = scp.interpolate.LinearNDInterpolator(x, y)(x)
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_interpolate(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)

        x_test = xp.array(
            [(-0.32, 0.022), (-0.22, 0.349), (0.297, 0.451), (0.237, 0.260)],
            dtype=xp.float64)

        yi = scp.interpolate.LinearNDInterpolator(x, y)(x_test)
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoketest_alternate(self, xp, scp):
        # Test at single points, alternate calling convention
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)

        yi = scp.interpolate.LinearNDInterpolator(
            (x[:, 0], x[:, 1]), y)(x[:, 0], x[:, 1])
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_smoketest(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j * y

        yi = scp.interpolate.LinearNDInterpolator(x, y)(x)
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tri_input(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j * y

        tri = scp.spatial.Delaunay(x)
        yi = scp.interpolate.LinearNDInterpolator(tri, y)(x)
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=0.5)
    def test_square(self, xp, scp):
        # Test barycentric interpolation on a square against a manual
        # implementation

        points = xp.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=xp.float64)
        values = xp.array([1., 2., -3., 5.], dtype=xp.float64)

        xx, yy = xp.broadcast_arrays(xp.linspace(0, 1, 14)[:, None],
                                     xp.linspace(0, 1, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()

        xi = xp.array([xx, yy]).T.copy()
        zi = scp.interpolate.LinearNDInterpolator(points, values)(xi)
        return zi

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1.0)
    def test_square_rescale(self, xp, scp):
        # Test barycentric interpolation on a rectangle with rescaling
        # agaings the same implementation without rescaling

        points = xp.array([(0, 0), (0, 100), (10, 100),
                          (10, 0)], dtype=xp.float64)
        values = xp.array([1., 2., -3., 5.], dtype=xp.float64)

        xx, yy = xp.broadcast_arrays(xp.linspace(0, 10, 14)[:, None],
                                     xp.linspace(0, 100, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = xp.array([xx, yy]).T.copy()
        zi = scp.interpolate.LinearNDInterpolator(
            points, values, rescale=True)(xi)
        return zi
