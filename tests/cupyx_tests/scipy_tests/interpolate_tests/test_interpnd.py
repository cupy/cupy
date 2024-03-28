
import cupy
from cupy import testing
import cupyx.scipy.interpolate  # NOQA
import cupyx.scipy.spatial  # NOQA

import numpy as np

try:
    import scipy.interpolate  # NOQA
except ImportError:
    pass

try:
    import scipy.spatial  # NOQA
except ImportError:
    pass

import pytest


class TestLinearNDInterpolator:
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

        x_test = xp.array(
            [(-0.32, 0.022), (-0.22, 0.349), (0.297, 0.451), (0.237, 0.260)],
            dtype=xp.float64)

        yi = scp.interpolate.LinearNDInterpolator(
            (x[:, 0], x[:, 1]), y)(x_test[:, 0], x_test[:, 1])
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_smoketest(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j * y

        x_test = xp.array(
            [(-0.32, 0.022), (-0.22, 0.349), (0.297, 0.451), (0.237, 0.260)],
            dtype=xp.float64)

        yi = scp.interpolate.LinearNDInterpolator(x, y)(x_test)
        return yi

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tri_input(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j * y

        x_test = xp.array(
            [(-0.32, 0.022), (-0.22, 0.349), (0.297, 0.451), (0.237, 0.260)],
            dtype=xp.float64)

        tri = scp.spatial.Delaunay(x)
        yi = scp.interpolate.LinearNDInterpolator(tri, y)(x_test)
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


class TestEstimateGradients2DGlobal:
    @pytest.mark.parametrize('func', [
        (lambda x, y: 0*x + 1),
        (lambda x, y: 0 + x),
        (lambda x, y: -2 + y),
        (lambda x, y: 3 + 3*x + 14.15*y)
    ])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1.0, atol=1e-6)
    def test_smoketest(self, func, xp, scp):
        x = xp.array([(0, 0), (0, 2),
                      (1, 0), (1, 2),
                      (0.25, 0.75),
                      (0.6, 0.8)], dtype=xp.float64)

        tri = scp.spatial.Delaunay(x)
        z = func(x[:, 0], x[:, 1])
        if xp is cupy:
            grad_fn = scp.interpolate._interpnd.estimate_gradients_2d_global
        else:
            grad_fn = scp.interpolate.interpnd.estimate_gradients_2d_global

        dz = grad_fn(tri, z, tol=1e-6)
        return dz


def compute_random_points(tri_points, xp):
    bary = xp.empty((tri_points.shape[0], 3), xp.float64)
    s = testing.shaped_random((tri_points.shape[0],), np, np.float64,
                              scale=1.0)
    t = testing.shaped_random((tri_points.shape[0],), np, np.float64,
                              scale=1-s)
    bary[:, 0] = xp.asarray(s)
    bary[:, 1] = xp.asarray(t)
    bary[:, 2] = xp.asarray(1 - s - t)

    return (xp.expand_dims(bary, -1) * tri_points).sum(1)


class TestCloughTocher2DInterpolator:

    def _check_accuracy(self, xp, scp, func, x=None, tol=1e-6, alternate=False,
                        rescale=False, **kw):
        if x is None:
            x = xp.array([(0, 0), (0, 1),
                          (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8),
                          (0.5, 0.2)],
                         dtype=xp.float64)

        if not alternate:
            ip = scp.interpolate.CloughTocher2DInterpolator(
                x, func(x[:, 0], x[:, 1]), tol=1e-6, rescale=rescale)
        else:
            ip = scp.interpolate.CloughTocher2DInterpolator(
                (x[:, 0], x[:, 1]), func(x[:, 0], x[:, 1]),
                tol=1e-6, rescale=rescale)

        p = testing.shaped_random((50, 2), xp, cupy.float64, 1.0, 1234)

        if not alternate:
            a = ip(p)
        else:
            a = ip(p[:, 0], p[:, 1])
        return a

    @pytest.mark.parametrize('func', [
        lambda x, y: 0*x + 1,
        lambda x, y: 0 + x,
        lambda x, y: -2 + y,
        lambda x, y: 3 + 3*x + 14.15*y,
    ])
    @pytest.mark.parametrize('alternate', [False, True])
    @pytest.mark.parametrize('rescale', [False, True])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1.0, atol=1e-6)
    def test_linear_smoketest(self, func, alternate, rescale, xp, scp):
        # Should be exact for linear functions, independent of triangulation
        return self._check_accuracy(
            xp, scp, func, tol=1e-13, atol=1e-7, rtol=1e-7,
            alternate=alternate, rescale=rescale)

    @pytest.mark.parametrize('func', [
        lambda x, y: x**2,
        lambda x, y: y**2,
        lambda x, y: x**2 - y**2,
        lambda x, y: x*y,
    ])
    @pytest.mark.parametrize('rescale', [False, True])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1.0, atol=1e-6)
    def test_quadratic_smoketest(self, func, rescale, xp, scp):
        # Should be reasonably accurate for quadratic functions
        return self._check_accuracy(xp, scp, func, tol=1e-9, rescale=rescale)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1.0, atol=1e-6)
    def test_tri_input(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5),
                      (0.5, 0.5), (0.25, 0.3)],
                     dtype=xp.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j*y

        tri = scp.spatial.Delaunay(x)
        yi = scp.interpolate.CloughTocher2DInterpolator(tri, y)(x)
        return yi

    @pytest.mark.parametrize('xp,scp', [(cupy, cupyx.scipy), (np, scipy)])
    def test_tri_input_rescale(self, xp, scp):
        # Test at single points
        x = xp.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)],
                     dtype=np.float64)
        y = xp.arange(x.shape[0], dtype=xp.float64)
        y = y - 3j*y

        tri = scp.spatial.Delaunay(x)
        match = ("Rescaling is not supported when passing a "
                 "Delaunay triangulation as ``points``.")
        with pytest.raises(ValueError, match=match):
            scp.interpolate.CloughTocher2DInterpolator(tri, y, rescale=True)(x)

    @pytest.mark.parametrize('func', [
        lambda x, y: x**2,
        lambda x, y: y**2,
        lambda x, y: x**2 - y**2,
        lambda x, y: x*y,
        lambda x, y: np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    ])
    @pytest.mark.parametrize('rescale', [False, True])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1.0, atol=1e-6)
    def test_dense(self, func, rescale, xp, scp):
        # Should be more accurate for dense meshes
        grid = xp.r_[xp.array([(0, 0), (0, 1), (1, 0), (1, 1)],
                              dtype=xp.float64),
                     testing.shaped_random(
                         (30 * 30, 2), xp, cupy.float64, 1, 4321)]

        return self._check_accuracy(xp, scp, func, x=grid, tol=1e-9)

    @pytest.mark.parametrize('xp,scp', [(cupy, cupyx.scipy), (np, scipy)])
    def test_wrong_ndim(self, xp, scp):
        x = testing.shaped_random((30, 3), xp, xp.float64)
        y = testing.shaped_random((30,), xp, xp.float64)
        with pytest.raises(ValueError):
            scp.interpolate.CloughTocher2DInterpolator(x, y)
