import pytest
import cupy as cp

from cupy.testing import (assert_allclose, assert_array_equal,
                          assert_array_almost_equal)
from cupy_backends.cuda.api import runtime
from pytest import raises as assert_raises

from cupyx.scipy.interpolate import RegularGridInterpolator, interpn

methods = ['linear', 'nearest']
if not runtime.is_hip:
    methods += ["slinear", "cubic", "quintic", 'pchip']

parametrize_rgi_interp_methods = pytest.mark.parametrize("method", methods)


class TestRegularGridInterpolator:
    def _get_sample_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 4
        values = cp.asarray([0., .5, 1.])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_2(self):
        # create another 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = cp.asarray([0., .5, 1.])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_3(self):
        # create another 4-D grid of 7 points in each dimension
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)] * 4
        values = cp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_4(self):
        # create another 4-D grid of 2 points in each dimension
        points = [(0.0, 1.0)] * 4
        values = cp.asarray([0.0, 1.0])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        points, values = self._get_sample_4d_3()
        values = values - 2j*values
        sample = cp.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        interp = RegularGridInterpolator(points, values, method=method)
        rinterp = RegularGridInterpolator(points, values.real, method=method)
        iinterp = RegularGridInterpolator(points, values.imag, method=method)

        v1 = interp(sample)
        v2 = rinterp(sample) + 1j*iinterp(sample)
        assert_allclose(v1, v2)

    def test_linear_xi1d(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([0.1, 0.1, 10., 9.])
        wanted = 1001.1
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_xi3d(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = cp.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    @pytest.mark.parametrize(
        "sample, wanted",
        [
            (cp.asarray([0.1, 0.1, 0.9, 0.9]), 1100.0),
            (cp.asarray([0.1, 0.1, 0.1, 0.1]), 0.0),
            (cp.asarray([0.0, 0.0, 0.0, 0.0]), 0.0),
            (cp.asarray([1.0, 1.0, 1.0, 1.0]), 1111.0),
            (cp.asarray([0.1, 0.4, 0.6, 0.9]), 1055.0),
        ],
    )
    def test_nearest(self, sample, wanted):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_edges(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        wanted = cp.asarray([0., 1111.])
        assert_array_almost_equal(interp(sample), wanted)

    def test_valid_create(self):
        # create a 2-D grid of 3 points in each dimension
        points = [(0., .5, 1.), (0., 1., .5)]
        values = cp.asarray([0., .5, 1.])
        values0 = values[:, cp.newaxis]
        values1 = values[cp.newaxis, :]
        values = (values0 + values1 * 10)
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [((0., .5, 1.), ), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, .75, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values,
                      method="undefmethod")

    def test_valid_call(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        assert_raises(ValueError, interp, sample, "undefmethod")
        sample = cp.asarray([[0., 0., 0.], [1., 1., 1.]])
        assert_raises(ValueError, interp, sample)
        sample = cp.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.1]])
        assert_raises(ValueError, interp, sample)

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = cp.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = cp.asarray([0., 1111., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = cp.asarray([-111.1, 1222.1, -11068., -1186.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = cp.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = cp.asarray([0., 11., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = cp.asarray([-12.1, 133.1, -1069., -97.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_fill(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=cp.nan)
        sample = cp.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [2.1, 2.1, -1.1, -1.1]])
        wanted = cp.asarray([cp.nan, cp.nan, cp.nan])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        assert_array_almost_equal(interp(sample, method="linear"), wanted)
        sample = cp.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = cp.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_invalid_fill_value(self):
        cp.random.seed(1234)
        x = cp.linspace(0, 2, 5)
        y = cp.linspace(0, 1, 7)
        values = cp.random.rand(5, 7)

        # integers can be cast to floats
        RegularGridInterpolator((x, y), values, fill_value=1)

        # complex values cannot
        assert_raises(ValueError, RegularGridInterpolator,
                      (x, y), values, fill_value=1+2j)

    def test_fillvalue_type(self):
        # from #3703; test that interpolator object construction succeeds
        values = cp.ones((10, 20, 30), dtype='>f4')
        points = [cp.arange(n) for n in values.shape]
        # xi = [(1, 1, 1)]
        RegularGridInterpolator(points, values)
        RegularGridInterpolator(points, values, fill_value=0.)

    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.
        def f(x, y):
            return x + y
        x = cp.linspace(1, 1, 1)
        y = cp.linspace(1, 10, 10)
        data = f(*cp.meshgrid(x, y, indexing="ij", sparse=True))

        interp = RegularGridInterpolator((x, y), data, method="linear",
                                         bounds_error=False, fill_value=101)

        # check values at the grid
        assert_allclose(interp(cp.array([[1, 1], [1, 5], [1, 10]])),
                        [2, 6, 11],
                        atol=1e-14)

        # check off-grid interpolation is indeed linear
        assert_allclose(interp(cp.array([[1, 1.4], [1, 5.3], [1, 10]])),
                        [2.4, 6.3, 11],
                        atol=1e-14)

        # check exrapolation w/ fill_value
        assert_allclose(interp(cp.array([1.1, 2.4])),
                        interp.fill_value,
                        atol=1e-14)

        # check extrapolation: linear along the `y` axis, const along `x`
        interp.fill_value = None
        assert_allclose(interp([[1, 0.3], [1, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        assert_allclose(interp([[1.5, 0.3], [1.9, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        # extrapolation with method='nearest'
        interp = RegularGridInterpolator((x, y), data, method="nearest",
                                         bounds_error=False, fill_value=None)
        assert_allclose(interp([[1.5, 1.8], [-4, 5.1]]),
                        [3, 6],
                        atol=1e-15)

    @pytest.mark.parametrize("fill_value", [None, cp.nan, cp.pi])
    @pytest.mark.parametrize("method", ['linear', 'nearest'])
    def test_length_one_axis2(self, fill_value, method):
        options = {"fill_value": fill_value, "bounds_error": False,
                   "method": method}

        x = cp.linspace(0, 2*cp.pi, 20)
        z = cp.sin(x)

        fa = RegularGridInterpolator((x,), z[:], **options)
        fb = RegularGridInterpolator((x, [0]), z[:, None], **options)

        x1a = cp.linspace(-1, 2*cp.pi+1, 100)
        za = fa(x1a)

        # evaluated at provided y-value, fb should behave exactly as fa
        y1b = cp.zeros(100)
        zb = fb(cp.vstack([x1a, y1b]).T)
        assert_allclose(zb, za)

        # evaluated at a different y-value, fb should return fill value
        y1b = cp.ones(100)
        zb = fb(cp.vstack([x1a, y1b]).T)
        if fill_value is None:
            assert_allclose(zb, za)
        else:
            assert_allclose(zb, fill_value)

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    def test_nan_x_1d(self, method):
        # gh-6624 : if x is nan, result should be nan
        f = RegularGridInterpolator((cp.array([1, 2, 3]),),
                                    cp.array([10, 20, 30]), fill_value=1,
                                    bounds_error=False, method=method)
        assert cp.isnan(f([cp.nan]))

        # test arbitrary nan pattern
        rng = cp.random.default_rng(8143215468)
        x = rng.random(size=100)*4
        i = rng.random(size=100) > 0.5
        x[i] = cp.nan

        # out-of-bounds comparisons, `out_of_bounds += x < grid[0]`,
        # generate numpy warnings if `x` contains nans.
        # These warnings should propagate to user
        res = f(x)

        assert_array_equal(res[i], cp.nan)
        assert_array_equal(res[~i], f(x[~i]))

        # also test the length-one axis f(nan)
        x = [1, 2, 3]
        y = [1, ]
        data = cp.ones((3, 1))
        f = RegularGridInterpolator((x, y), data, fill_value=1,
                                    bounds_error=False, method=method)
        assert cp.isnan(f([cp.nan, 1]))
        assert cp.isnan(f([1, cp.nan]))

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    def test_nan_x_2d(self, method):
        x, y = cp.array([0, 1, 2]), cp.array([1, 3, 7])

        def f(x, y):
            return x**2 + y**2

        xg, yg = cp.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)
        interp = RegularGridInterpolator((x, y), data,
                                         method=method, bounds_error=False)

        res = interp([[1.5, cp.nan], [1, 1]])
        assert_allclose(res[1], 2, atol=1e-14)
        assert cp.isnan(res[0])

        # test arbitrary nan pattern
        rng = cp.random.default_rng(8143215468)
        x = rng.random(size=100)*4-1
        y = rng.random(size=100)*8
        i1 = rng.random(size=100) > 0.5
        i2 = rng.random(size=100) > 0.5
        i = i1 | i2
        x[i1] = cp.nan
        y[i2] = cp.nan
        z = cp.array([x, y]).T
        # out-of-bounds comparisons, `out_of_bounds += x < grid[0]`,
        # generate numpy warnings if `x` contains nans.
        # These warnings should propagate to user (since `x` is user
        # input)
        res = interp(z)

        assert_array_equal(res[i], cp.nan)
        assert_array_equal(res[~i], interp(z[~i]))

    @parametrize_rgi_interp_methods
    def test_descending_points(self, method):
        def val_func_3d(x, y, z):
            return 2 * x ** 3 + 3 * y ** 2 - z

        x = cp.linspace(1, 4, 11)
        y = cp.linspace(4, 7, 22)
        z = cp.linspace(7, 9, 33)
        points = (x, y, z)
        values = val_func_3d(
            *cp.meshgrid(*points, indexing='ij', sparse=True))
        my_interpolating_function = RegularGridInterpolator(points,
                                                            values,
                                                            method=method)
        pts = cp.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
        correct_result = my_interpolating_function(pts)

        # descending data
        x_descending = x[::-1]
        y_descending = y[::-1]
        z_descending = z[::-1]
        points_shuffled = (x_descending, y_descending, z_descending)
        values_shuffled = val_func_3d(
            *cp.meshgrid(*points_shuffled, indexing='ij', sparse=True))
        my_interpolating_function = RegularGridInterpolator(
            points_shuffled, values_shuffled, method=method)
        test_result = my_interpolating_function(pts)

        assert_array_equal(correct_result, test_result)

    def test_invalid_points_order(self):
        def val_func_2d(x, y):
            return 2 * x ** 3 + 3 * y ** 2

        x = cp.array([.5, 2., 0., 4., 5.5])  # not ascending or descending
        y = cp.array([.5, 2., 3., 4., 5.5])
        points = (x, y)
        values = val_func_2d(*cp.meshgrid(*points, indexing='ij',
                                          sparse=True))
        match = "must be strictly ascending or descending"
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values)

    @parametrize_rgi_interp_methods
    def test_fill_value(self, method):
        interp = RegularGridInterpolator([cp.arange(6)], cp.ones(6),
                                         method=method, bounds_error=False)
        assert cp.isnan(interp([10]))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        # Verify that non-scalar valued values also works
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        rng = cp.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)
        assert_array_equal(v.shape, (7, 3, 8), err_msg=method)

        vs = []
        for j in range(8):
            interp = RegularGridInterpolator(points, values[..., j],
                                             method=method,
                                             bounds_error=False)
            vs.append(interp(sample))
        v2 = cp.array(vs).transpose(1, 2, 0)

        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize("flip_points", [False, True])
    def test_nonscalar_values_2(self, method, flip_points):
        # Verify that non-scalar valued values also work : use different
        # lengths of axes to simplify tracing the internals
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        # verify, that strictly decreasing dimensions work
        if flip_points:
            points = [tuple(reversed(p)) for p in points]

        rng = cp.random.default_rng(1234)

        trailing_points = (3, 2)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)   # a single sample point !

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = cp.empty((values.shape[-2:]))
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                vs[i, j] = interp(sample)
        v2 = cp.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    def test_nonscalar_values_linear_2D(self):
        # Verify that non-scalar values work in the 2D fast path
        method = 'linear'
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), ]

        rng = cp.random.default_rng(1234)

        trailing_points = (3, 4)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, *trailing_points))
        sample = rng.random(2)   # a single sample point !

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = cp.empty((values.shape[-2:]))
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                vs[i, j] = interp(sample)
        v2 = cp.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)


class MyValue:
    """
    Minimal indexable object
    """

    def __init__(self, shape):
        self.ndim = 2
        self.shape = shape
        self._v = cp.arange(cp.prod(cp.array(shape))).reshape(shape)

    def __getitem__(self, idx):
        return self._v[idx]

    def __array_interface__(self):
        return None

    def __array__(self):
        raise RuntimeError("No array representation")


class TestInterpN:
    def _sample_2d_data(self):
        x = cp.array([.5, 2., 3., 4., 5.5, 6.])
        y = cp.array([.5, 2., 3., 4., 5.5, 6.])
        z = cp.array(
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

    def _sample_4d_data(self):
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = cp.asarray([0., .5, 1.])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def test_linear_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="linear")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_linear_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = cp.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="linear",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_nearest_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values, method="nearest")
        sample = cp.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="nearest")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_nearest_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = cp.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="nearest",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_xi_1d(self):
        # verify that 1-D xi works as expected
        points, values = self._sample_4d_data()
        sample = cp.asarray([0.1, 0.1, 10., 9.])
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None, :], bounds_error=False)
        assert_allclose(v1, v2)

    def test_xi_nd(self):
        # verify that higher-d xi works as expected
        points, values = self._sample_4d_data()

        cp.random.seed(1234)
        sample = cp.random.rand(2, 3, 4)

        v1 = interpn(points, values, sample, method='nearest',
                     bounds_error=False)
        assert_array_equal(v1.shape, (2, 3))

        v2 = interpn(points, values, sample.reshape(-1, 4),
                     method='nearest', bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_xi_broadcast(self, method):
        # verify that the interpolators broadcast xi
        x, y, values = self._sample_2d_data()
        points = (x, y)

        xi = cp.linspace(0, 1, 2)
        yi = cp.linspace(0, 3, 3)

        sample = (xi[:, None], yi[None, :])
        v1 = interpn(points, values, sample, method=method, bounds_error=False)
        assert_array_equal(v1.shape, (2, 3))

        xx, yy = cp.meshgrid(xi, yi)
        sample = cp.c_[xx.T.ravel(), yy.T.ravel()]

        v2 = interpn(points, values, sample,
                     method=method, bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        # Verify that non-scalar valued values also works
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        rng = cp.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))

        v = interpn(points, values, sample, method=method,
                    bounds_error=False)
        assert_array_equal(v.shape, (7, 3, 8), err_msg=method)

        vs = [interpn(points, values[..., j], sample, method=method,
                      bounds_error=False) for j in range(8)]
        v2 = cp.array(vs).transpose(1, 2, 0)

        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    def test_nonscalar_values_2(self, method):
        # Verify that non-scalar valued values also work : use different
        # lengths of axes to simplify tracing the internals
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        rng = cp.random.default_rng(1234)

        trailing_points = (3, 2)
        # NB: values has a `num_trailing_dims` trailing dimension
        values = rng.random((6, 7, 8, 9, *trailing_points))
        sample = rng.random(4)   # a single sample point !

        v = interpn(points, values, sample, method=method, bounds_error=False)

        # v has a single sample point *per entry in the trailing dimensions*
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        vs = [[
            interpn(points, values[..., i, j], sample, method=method,
                    bounds_error=False) for i in range(values.shape[-2])
        ] for j in range(values.shape[-1])]

        assert_allclose(v, cp.asarray(vs).T, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j*values

        sample = cp.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        v1 = interpn(points, values, sample, method=method)
        v2r = interpn(points, values.real, sample, method=method)
        v2i = interpn(points, values.imag, sample, method=method)
        v2 = v2r + 1j*v2i
        assert_allclose(v1, v2)

    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.

        values = cp.array([[0.1, 1, 10]])
        xi = cp.array([[1, 2.2], [1, 3.2], [1, 3.8]])

        res = interpn(([1], [2, 3, 4]), values, xi)
        wanted = [0.9*0.2 + 0.1,   # on [2, 3) it's 0.9*(x-2) + 0.1
                  9*0.2 + 1,       # on [3, 4] it's 9*(x-3) + 1
                  9*0.8 + 1]

        assert_allclose(res, wanted, atol=1e-15)

        # check extrapolation
        xi = cp.array([[1.1, 2.2], [1.5, 3.2], [-2.3, 3.8]])
        res = interpn(([1], [2, 3, 4]), values, xi,
                      bounds_error=False, fill_value=None)

        assert_allclose(res, wanted, atol=1e-15)

    def test_descending_points(self):
        def value_func_4d(x, y, z, a):
            return 2 * x ** 3 + 3 * y ** 2 - z - a

        x1 = cp.array([0, 1, 2, 3])
        x2 = cp.array([0, 10, 20, 30])
        x3 = cp.array([0, 10, 20, 30])
        x4 = cp.array([0, .1, .2, .30])
        points = (x1, x2, x3, x4)
        values = value_func_4d(
            *cp.meshgrid(*points, indexing='ij', sparse=True))
        pts = (cp.array(0.1),
               cp.array(0.3),
               cp.transpose(cp.linspace(0, 30, 4)),
               cp.linspace(0, 0.3, 4))
        correct_result = interpn(points, values, pts)

        x1_descend = x1[::-1]
        x2_descend = x2[::-1]
        x3_descend = x3[::-1]
        x4_descend = x4[::-1]
        points_shuffled = (x1_descend, x2_descend, x3_descend, x4_descend)
        values_shuffled = value_func_4d(
            *cp.meshgrid(*points_shuffled, indexing='ij', sparse=True))
        test_result = interpn(points_shuffled, values_shuffled, pts)

        assert_array_equal(correct_result, test_result)

    def test_invalid_points_order(self):
        x = cp.array([.5, 2., 0., 4., 5.5])  # not ascending or descending
        y = cp.array([.5, 2., 3., 4., 5.5])
        z = cp.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        xi = cp.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T

        match = "must be strictly ascending or descending"
        with pytest.raises(ValueError, match=match):
            interpn((x, y), z, xi)

    def test_invalid_xi_dimensions(self):
        # https://github.com/scipy/scipy/issues/16519
        points = [cp.array((0, 1))]
        values = cp.array([0, 1])
        xi = cp.ones((1, 1, 3))
        msg = ("The requested sample points xi have dimension 3, but this "
               "RegularGridInterpolator has dimension 1")
        with assert_raises(ValueError, match=msg):
            interpn(points, values, xi)
