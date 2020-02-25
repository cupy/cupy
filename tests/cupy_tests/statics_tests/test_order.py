import unittest
import warnings

from cupy import testing


_all_interpolations = (
    'lower',
    'higher',
    'midpoint',
    # 'nearest', # TODO(hvy): Not implemented
    'linear')


def for_all_interpolations(name='interpolation'):
    return testing.for_orders(_all_interpolations, name=name)


@testing.gpu
class TestOrder(unittest.TestCase):

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_defaults(self, xp, dtype, interpolation):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = testing.shaped_random((3,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_no_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((10, 2, 4, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=None, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_neg_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((4, 3, 10, 2, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=-1, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_tuple_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((1, 6, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=(0, 1, 2), interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_scalar_q(self, xp, dtype, interpolation):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = 13.37
        return xp.percentile(a, q, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_percentile_keepdims(self, xp, dtype, interpolation):
        a = testing.shaped_random((7, 2, 9, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(
            a, q, axis=None, keepdims=True, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_float_dtypes(no_float16=True)  # NumPy raises error on int8
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_out(self, xp, dtype, interpolation):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.percentile(
            a, q, axis=-1, interpolation=interpolation, out=out)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_percentile_bad_q(self, xp, dtype, interpolation):
        a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=-1, interpolation=interpolation)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_percentile_unexpected_interpolation(self, xp, dtype):
        a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=-1, interpolation='deadbeef')

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmax(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_all_nan(self, xp, dtype):
        a = xp.array([float('nan'), float('nan')], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = xp.nanmax(a)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, RuntimeWarning)
        return m

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmin(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_all_nan(self, xp, dtype):
        a = xp.array([float('nan'), float('nan')], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = xp.nanmin(a)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, RuntimeWarning)
        return m

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.ptp(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.ptp(a, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        return xp.ptp(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_all_nan(self, xp, dtype):
        a = xp.array([float('nan'), float('nan')], dtype)
        return xp.ptp(a)
