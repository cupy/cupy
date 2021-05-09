import unittest
import warnings

import numpy
import pytest

import cupy
import cupy._core._accelerator as _acc
from cupy import cuda
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
    @testing.numpy_cupy_allclose()
    def test_percentile_q_list(self, xp, dtype, interpolation):
        a = testing.shaped_arange((1001,), xp, dtype)
        q = [99, 99.9]
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
    def test_percentile_bad_q(self, dtype, interpolation):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=100)
            with pytest.raises(ValueError):
                xp.percentile(a, q, axis=-1, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_percentile_out_of_range_q(self, dtype, interpolation):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            for q in [[-0.1], [100.1]]:
                with pytest.raises(ValueError):
                    xp.percentile(a, q, axis=-1, interpolation=interpolation)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_percentile_unexpected_interpolation(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
            with pytest.raises(ValueError):
                xp.percentile(a, q, axis=-1, interpolation='deadbeef')

    # See gh-4453
    @testing.for_float_dtypes()
    def test_percentile_memory_access(self, dtype):
        # Create an allocator that guarantees array allocated in
        # cupy.percentile call will be followed by a NaN
        original_allocator = cuda.get_allocator()

        def controlled_allocator(size):
            memptr = original_allocator(size)
            base_size = memptr.mem.size
            assert base_size % 512 == 0
            item_size = dtype().itemsize
            shape = (base_size // item_size,)
            x = cupy.ndarray(memptr=memptr, shape=shape, dtype=dtype)
            x.fill(cupy.nan)
            return memptr

        # Check that percentile still returns non-NaN results
        a = testing.shaped_random((5,), cupy, dtype)
        q = cupy.array((0, 100), dtype=dtype)

        cuda.set_allocator(controlled_allocator)
        try:
            percentiles = cupy.percentile(a, q, axis=None,
                                          interpolation='linear')
        finally:
            cuda.set_allocator(original_allocator)

        assert not cupy.any(cupy.isnan(percentiles))

    @testing.for_all_dtypes()
    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_quantile_defaults(self, xp, dtype, interpolation):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = testing.shaped_random((3,), xp, scale=1)
        return xp.quantile(a, q, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_quantile_q_list(self, xp, dtype, interpolation):
        a = testing.shaped_arange((1001,), xp, dtype)
        q = [.99, .999]
        return xp.quantile(a, q, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_quantile_no_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((10, 2, 4, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=None, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_neg_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((4, 3, 10, 2, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=-1, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_tuple_axis(self, xp, dtype, interpolation):
        a = testing.shaped_random((1, 6, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=(0, 1, 2), interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_quantile_scalar_q(self, xp, dtype, interpolation):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = .1337
        return xp.quantile(a, q, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_quantile_keepdims(self, xp, dtype, interpolation):
        a = testing.shaped_random((7, 2, 9, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(
            a, q, axis=None, keepdims=True, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_float_dtypes(no_float16=True)  # NumPy raises error on int8
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_out(self, xp, dtype, interpolation):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.quantile(
            a, q, axis=-1, interpolation=interpolation, out=out)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_bad_q(self, dtype, interpolation):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.quantile(a, q, axis=-1, interpolation=interpolation)

    @for_all_interpolations()
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_out_of_range_q(self, dtype, interpolation):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            for q in [[-0.1], [1.1]]:
                with pytest.raises(ValueError):
                    xp.quantile(a, q, axis=-1, interpolation=interpolation)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_unexpected_interpolation(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.quantile(a, q, axis=-1, interpolation='deadbeef')

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmax(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=1)

    @testing.for_all_dtypes()
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
        assert len(w) == 1
        assert w[0].category is RuntimeWarning
        return m

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmin(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=1)

    @testing.for_all_dtypes()
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
        assert len(w) == 1
        assert w[0].category is RuntimeWarning
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
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype)
        return xp.ptp(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_all_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), float('nan')], dtype)
        return xp.ptp(a)


# See gh-4607
# "Magic" values used in this test were empirically found to result in
# non-monotonicity for less accurate linear interpolation formulas
@testing.parameterize(*testing.product({
    'magic_value': (-29, -53, -207, -16373, -99999,)
}))
@testing.gpu
class TestPercentileMonotonic(unittest.TestCase):

    @testing.with_requires('numpy>=1.20')
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_monotonic(self, dtype, xp):
        a = testing.shaped_random((5,), xp, dtype)

        a[0] = self.magic_value
        a[1] = self.magic_value
        q = xp.linspace(0, 100, 21)
        percentiles = xp.percentile(a, q, interpolation='linear')

        # Assert that percentile output increases monotonically
        assert xp.all(xp.diff(percentiles) >= 0)

        return percentiles
