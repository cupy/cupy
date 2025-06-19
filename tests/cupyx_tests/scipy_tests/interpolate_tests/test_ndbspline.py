import numpy
import cupy
from cupy.cuda import runtime
from cupy import testing

import cupyx.scipy.interpolate  # NOQA

try:
    import scipy.interpolate  # NOQA
except ImportError:
    pass

import pytest
import itertools


@testing.with_requires('scipy>=1.12')
class TestNdBSpline:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_1D(self, xp, scp):
        # test ndim=1 agrees with BSpline
        n, k = 11, 3
        n_tr = 7
        t = testing.shaped_random((n + k + 1,), xp, xp.float64, 1.0, 1234)
        t = xp.sort(t)
        c = testing.shaped_random((n, n_tr), xp, xp.float64, 1.0)

        nb = scp.interpolate.NdBSpline((t,), c, k)
        xi = testing.shaped_random((21,), xp, xp.float64, 1.0, 2234)
        return nb(xi[:, None])

    def make_2d_case(self, xp, scp):
        # make a 2D separable spline
        x = xp.arange(6)
        y = x**3
        spl = scp.interpolate.make_interp_spline(x, y, k=3)

        y_1 = x**3 + 2*x
        spl_1 = scp.interpolate.make_interp_spline(x, y_1, k=3)

        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, 3

    def make_2d_mixed(self, xp, scp):
        # make a 2D separable spline w/ kx=3, ky=2
        x = xp.arange(6)
        y = x**3
        spl = scp.interpolate.make_interp_spline(x, y, k=3)

        x = xp.arange(5) + 1.5
        y_1 = x**2 + 2*x
        spl_1 = scp.interpolate.make_interp_spline(x, y_1, k=2)

        t2 = (spl.t, spl_1.t)
        c2 = spl.c[:, None] * spl_1.c[None, :]

        return t2, c2, spl.k, spl_1.k

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_separable(self, xp, scp):
        xi = xp.asarray([(1.5, 2.5), (2.5, 1), (0.5, 1.5)],
                        dtype=xp.float64)
        t2, c2, _ = self.make_2d_case(xp, scp)

        # check evaluation on a 2D array: the 1D array of 2D points
        bspl2 = scp.interpolate.NdBSpline(t2, c2, k=3)
        r1 = bspl2(xi)

        # now check on a multidim xi
        xi = testing.shaped_random((4, 3, 2), xp, xp.float64, 5.0, 12345)
        result = bspl2(xi)
        return r1, result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_separable_2(self, xp, scp):
        result = []

        # test `c` with trailing dimensions, i.e. c.ndim > ndim
        # ndim = 2
        # xi = xp.asarray([(1.5, 2.5), (2.5, 1), (0.5, 1.5)], dtype=xp.float64)
        # target = [x**3 * (y**3 + 2*y) for (x, y) in xi]

        t2, c2, k = self.make_2d_case(xp, scp)
        c2_4 = xp.dstack((c2, c2, c2, c2))   # c22.shape = (6, 6, 4)

        xy = xp.asarray([1.5, 2.5], dtype=xp.float64)
        bspl2_4 = scp.interpolate.NdBSpline(t2, c2_4, k=3)

        r1 = bspl2_4(xy)
        result.append(r1)

        val_single = scp.interpolate.NdBSpline(t2, c2, k)(xy)
        result.append(val_single)

        # two trailing dimensions
        c2_22 = c2_4.reshape((6, 6, 2, 2))
        bspl2_22 = scp.interpolate.NdBSpline(t2, c2_22, k=3)

        r2 = bspl2_22(xy)
        result.append(r2)

        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_random(self, xp, scp):
        k = 3

        x_v = xp.sort(testing.shaped_random((7,), xp, xp.float64, 3.0, 12345))
        y_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 54321))

        tx = xp.r_[0, 0, 0, 0, x_v, 3, 3, 3, 3]
        ty = xp.r_[0, 0, 0, 0, y_v, 4, 4, 4, 4]
        c = testing.shaped_random(
            (tx.size - k - 1, ty.size - k - 1), xp, xp.float64, 1.0)

        spl = scp.interpolate.NdBSpline((tx, ty), c, k=k)

        xi = xp.asarray([1., 1.], dtype=xp.float64)
        r1 = spl(xi)

        xi = xp.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]
        return r1, spl(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_mixed(self, xp, scp):
        t2, c2, kx, ky = self.make_2d_mixed(xp, scp)
        xi = xp.asarray([(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)], dtype=xp.float64)
        # target = [x**3 * (y**2 + 2*y) for (x, y) in xi]
        bspl2 = scp.interpolate.NdBSpline(t2, c2, k=(kx, ky))
        return bspl2(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_derivative(self, xp, scp):
        t2, c2, kx, ky = self.make_2d_mixed(xp, scp)
        xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
        bspl2 = scp.interpolate.NdBSpline(t2, c2, k=(kx, ky))

        result = []

        der = bspl2(xi, nu=(1, 0))
        result.append(der)

        der = bspl2(xi, nu=(1, 1))
        result.append(der)

        der = bspl2(xi, nu=(0, 0))
        result.append(der)

        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_mixed_random(self, xp, scp):
        kx, ky = 2, 3

        x_v = xp.sort(testing.shaped_random((7,), xp, xp.float64, 3.0, 12345))
        y_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 54321))

        tx = xp.r_[0, 0, 0, 0, x_v, 3, 3, 3, 3]
        ty = xp.r_[0, 0, 0, 0, y_v, 4, 4, 4, 4]
        c = testing.shaped_random(
            (tx.size - kx - 1, ty.size - ky - 1), xp, xp.float64, 1.0)

        xi = xp.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        bspl2 = scp.interpolate.NdBSpline((tx, ty), c, k=(kx, ky))
        return bspl2(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-12)
    def test_tx_neq_ty(self, xp, scp):
        # 2D separable spline w/ len(tx) != len(ty)
        x = xp.arange(6)
        y = xp.arange(7) + 1.5

        spl_x = scp.interpolate.make_interp_spline(x, x**3, k=3)
        spl_y = scp.interpolate.make_interp_spline(y, y**2 + 2*y, k=3)
        cc = spl_x.c[:, None] * spl_y.c[None, :]
        bspl = scp.interpolate.NdBSpline(
            (spl_x.t, spl_y.t), cc, (spl_x.k, spl_y.k))

        # values = (x**3)[:, None] * (y**2 + 2*y)[None, :]
        xi = [(a, b) for a, b in itertools.product(x, y)]
        bxi = bspl(xi)
        return bxi

    def make_3d_case(self, xp, scp):
        # make a 3D separable spline
        x = xp.arange(6)
        y = x**3
        spl = scp.interpolate.make_interp_spline(x, y, k=3)

        y_1 = x**3 + 2*x
        spl_1 = scp.interpolate.make_interp_spline(x, y_1, k=3)

        y_2 = x**3 + 3*x + 1
        spl_2 = scp.interpolate.make_interp_spline(x, y_2, k=3)

        t2 = (spl.t, spl_1.t, spl_2.t)
        c2 = (spl.c[:, None, None] *
              spl_1.c[None, :, None] *
              spl_2.c[None, None, :])

        return t2, c2, 3

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_3D_separable(self, xp, scp):
        x, y, z = testing.shaped_random((3, 11), xp, xp.float64, 5.0, 12345)
        # target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        t3, c3, _ = self.make_3d_case(xp, scp)
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3)

        xi = [_ for _ in zip(x, y, z)]
        result = bspl3(xi)
        return result

    @pytest.mark.parametrize('nu', [
        (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 3), (2, 1, 4)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_3D_derivative(self, nu, xp, scp):
        t3, c3, _ = self.make_3d_case(xp, scp)
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3)

        x, y, z = testing.shaped_random((3, 11), xp, xp.float64, 5.0, 12345)
        xi = [_ for _ in zip(x, y, z)]

        return bspl3(xi, nu=nu)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_3D_random(self, xp, scp):
        k = 3

        x_v = xp.sort(testing.shaped_random((7,), xp, xp.float64, 3.0, 12345))
        y_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 54321))
        z_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 15432))

        tx = xp.r_[0, 0, 0, 0, x_v, 3, 3, 3, 3]
        ty = xp.r_[0, 0, 0, 0, y_v, 4, 4, 4, 4]
        tz = xp.r_[0, 0, 0, 0, z_v, 4, 4, 4, 4]

        c = testing.shaped_random(
            (tx.size - k - 1, ty.size - k - 1, tz.size - k - 1),
            xp, xp.float64, 1.0)

        spl = scp.interpolate.NdBSpline((tx, ty, tz), c, k=k)

        xi = (1., 1., 1)
        r1 = spl(xi)

        xi = xp.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        return r1, spl(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_3D_random_complex(self, xp, scp):
        k = 3

        x_v = xp.sort(testing.shaped_random((7,), xp, xp.float64, 3.0, 12345))
        y_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 54321))
        z_v = xp.sort(testing.shaped_random((8,), xp, xp.float64, 4.0, 15432))

        tx = xp.r_[0, 0, 0, 0, x_v, 3, 3, 3, 3]
        ty = xp.r_[0, 0, 0, 0, y_v, 4, 4, 4, 4]
        tz = xp.r_[0, 0, 0, 0, z_v, 4, 4, 4, 4]

        cr = testing.shaped_random(
            (tx.size - k - 1, ty.size - k - 1, tz.size - k - 1),
            xp, xp.float64, 1.0)
        ci = testing.shaped_random(
            (tx.size - k - 1, ty.size - k - 1, tz.size - k - 1),
            xp, xp.float64, 1.0, 5689)
        c = cr + 1j * ci

        spl = scp.interpolate.NdBSpline((tx, ty, tz), c, k=k)

        xi = xp.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1],
                   [0.9, 1.4, 1.9]]
        return spl(xi)

    @pytest.mark.parametrize('cls_extrap', [None, True])
    @pytest.mark.parametrize('call_extrap', [None, True])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extrapolate_3D_separable(self, cls_extrap, call_extrap, xp, scp):
        # test that extrapolate=True does extrapolate
        t3, c3, k = self.make_3d_case(xp, scp)
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        x, y, z = map(xp.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]

        result = bspl3(xi, extrapolate=call_extrap)
        return result

    @pytest.mark.parametrize('extrap', [(False, True), (True, None)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extrapolate_3D_separable_2(self, extrap, xp, scp):
        # test that call(..., extrapolate=None) defers to self.extrapolate,
        # otherwise supersedes self.extrapolate
        t3, c3, k = self.make_3d_case(xp, scp)
        cls_extrap, call_extrap = extrap
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)

        # evaluate out of bounds
        x, y, z = [-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5]
        x, y, z = map(xp.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        # target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi, extrapolate=call_extrap)
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extrapolate_false_3D_separable(self, xp, scp):
        # test that extrapolate=False produces nans for out-of-bounds values
        t3, c3, _ = self.make_3d_case(xp, scp)
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        x, y, z = [-2, 1, 7], [-3, 0.5, 6.5], [-1, 1.5, 7.5]
        x, y, z = map(xp.asarray, (x, y, z))
        xi = [_ for _ in zip(x, y, z)]
        # target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi, extrapolate=False)
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_x_nan_3D(self, xp, scp):
        # test that spline(nan) is nan
        t3, c3, _ = self.make_3d_case(xp, scp)
        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3)

        # evaluate out of bounds and inside
        x = xp.asarray([-2, 3, xp.nan, 1, 2, 7, xp.nan])
        y = xp.asarray([-3, 3.5, 1, xp.nan, 3, 6.5, 6.5])
        z = xp.asarray([-1, 3.5, 2, 3, xp.nan, 7.5, 7.5])
        xi = [_ for _ in zip(x, y, z)]
        # target = x**3 * (y**3 + 2*y) * (z**3 + 3*z + 1)

        result = bspl3(xi)
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_non_c_contiguous(self, xp, scp):
        # check that non C-contiguous inputs are OK
        kx, ky = 3, 3
        tx = xp.sort(testing.shaped_random((16,), xp, xp.float64, 4.0, 12345))
        tx = xp.r_[(tx[0],) * kx, tx, (tx[-1],) * kx]
        ty = xp.sort(testing.shaped_random((16,), xp, xp.float64, 4.0, 54321))
        ty = xp.r_[(ty[0],) * ky, ty, (ty[-1],) * ky]

        assert not tx[::2].flags.c_contiguous
        assert not ty[::2].flags.c_contiguous

        c = testing.shaped_random(
            (tx.size//2 - kx - 1, ty.size//2 - ky - 1), xp, xp.float64, 1.0)
        c = c.T
        assert not c.flags.c_contiguous

        xi = xp.c_[[1, 1.5, 2],
                   [1.1, 1.6, 2.1]]

        bspl2 = scp.interpolate.NdBSpline((tx[::2], ty[::2]), c, k=(kx, ky))
        return bspl2(xi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_readonly(self, xp, scp):
        t3, c3, _ = self.make_3d_case(xp, scp)
        for i in range(3):
            t3[i].flags.writeable = False
        c3.flags.writeable = False

        bspl3 = scp.interpolate.NdBSpline(t3, c3, k=3)
        return bspl3((1, 2, 3))

    @testing.with_requires('scipy>=1.13')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_design_matrix(self, xp, scp):
        t3, c3, k = self.make_3d_case(xp, scp)

        xi = xp.asarray([[1, 2, 3], [4, 5, 6]])
        dm = scp.interpolate.NdBSpline(t3, c3, k).design_matrix(xi, t3, k)
        dm1 = scp.interpolate.NdBSpline.design_matrix(xi, t3, [k, k, k])

        with pytest.raises(ValueError):
            scp.interpolate.NdBSpline.design_matrix([[1, 2]], t3, [k]*3)

        return dm.todense(), dm1.todense()

    @testing.with_requires('scipy>=1.13')
    @pytest.mark.parametrize('k', [(3, 1), (1, 3), (1, 1), (3, 3)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_design_matrix_2(self, xp, scp, k):
        xvals = [(a, b)
                 for a, b in
                 itertools.product(xp.arange(6), xp.arange(7) + 1.5)
                 ]
        xvals = xp.asarray(xvals)

        t = (xp.array([0., 0., 0., 0., 2., 3., 5., 5., 5., 5.]),
             xp.array([1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 7.5])
             )

        dm = scp.interpolate.NdBSpline.design_matrix(xvals, t, k)
        return dm.todense()


@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
@testing.with_requires('scipy>=1.13')
class TestMakeND:

    def _make(self, xp):
        if xp == cupy:
            return cupyx.scipy.interpolate._ndbspline.make_ndbspl
        else:
            import functools
            import scipy.sparse.linalg as ssl
            return functools.partial(scipy.interpolate._ndbspline.make_ndbspl,
                                     solver=ssl.spsolve)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_separable_simple(self, xp, scp):
        x = xp.arange(6)
        y = xp.arange(6) + 0.5
        values = x[:, None]**3 * (y**3 + 2*y)[None, :]
        xi = [(a, b) for a, b in itertools.product(x, y)]

        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y), values, k=1)
        return bspl(xi), bspl.c

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-11)
    def test_2D_separable_trailing_dims(self, xp, scp):
        # test `c` with trailing dimensions, i.e. c.ndim > ndim
        x = xp.arange(6)
        y = xp.arange(6)
        xi = [(a, b) for a, b in itertools.product(x, y)]

        # make values4.shape = (6, 6, 4)
        values = x[:, None]**3 * (y**3 + 2*y)[None, :]
        values4 = xp.dstack((values, values, values, values))

        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y), values4, k=3)
        return bspl(xi), bspl.c

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-13)
    def test_2D_separable_trailing_dims_2(self, xp, scp):
        x = xp.arange(6)
        y = xp.arange(6)
        xi = [(a, b) for a, b in itertools.product(x, y)]

        # make values4.shape = (6, 6, 4)
        values = x[:, None]**3 * (y**3 + 2*y)[None, :]
        values4 = xp.dstack((values, values, values, values))

        # now two trailing dimensions
        values22 = values4.reshape((6, 6, 2, 2))

        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y), values22, k=3)
        return bspl(xi)

    @pytest.mark.parametrize('k', [(3, 3), (1, 1), (3, 1), (1, 3), (3, 5)])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_2D_mixed(self, k, xp, scp):
        # make a 2D separable spline w/ len(tx) != len(ty)
        x = xp.arange(6)
        y = xp.arange(7) + 1.5
        xi = [(a, b) for a, b in itertools.product(x, y)]
        values = (x**3)[:, None] * (y**2 + 2*y)[None, :]
        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y), values, k=k)
        return bspl(xi)

    def _get_sample_2d_data(self, xp):
        # from test_rgi.py::TestIntepN
        x = xp.array([.5, 2., 3., 4., 5.5, 6.])
        y = xp.array([.5, 2., 3., 4., 5.5, 6.])
        z = xp.array(
            [
                [1, 2, 1, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 3, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
            ]
        )
        return x, y, z

    @pytest.mark.parametrize('k', [1, 3, 5])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_2D_vs_RGI_k(self, xp, scp, k):
        x, y, z = self._get_sample_2d_data(xp)
        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y), z, k=1)
        xi = xp.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        return bspl(xi)

    @pytest.mark.parametrize(
        'k, meth', [(1, 'linear'), (3, 'cubic_legacy'), (5, 'quintic_legacy')]
    )
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_3D_random_vs_RGI(self, k, meth, xp, scp):
        rndm = numpy.random.default_rng(123456)
        x = xp.cumsum(xp.asarray(rndm.uniform(size=6)))
        y = xp.cumsum(xp.asarray(rndm.uniform(size=7)))
        z = xp.cumsum(xp.asarray(rndm.uniform(size=8)))
        values = xp.asarray(rndm.uniform(size=(6, 7, 8)))

        make_ndbspl = self._make(xp)
        bspl = make_ndbspl((x, y, z), values, k=k)

        xi = rndm.uniform(low=0.7, high=2.1, size=(11, 3))
        xi = xp.asarray(xi)
        return bspl(xi)
