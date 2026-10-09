from __future__ import annotations

import warnings

import numpy
import pytest

import cupy
import cupy._core._accelerator as _acc
from cupy import cuda
from cupy import testing
from cupy._statistics import order as order_module


_all_methods = (
    "inverted_cdf",
    # 'averaged_inverted_cdf',      # TODO(takagi) Not implemented
    # 'closest_observation',        # TODO(takagi) Not implemented
    # 'interpolated_inverted_cdf',  # TODO(takagi) Not implemented
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)


@pytest.fixture
def _fix_gamma(monkeypatch):
    if numpy.__version__ == "2.4.1":
        # NumPy 2.4.0 had a surprisingly large change, but I (seberg)
        # incorrectly undid the change, making things maybe worse...
        # this fixes that...
        # See also https://github.com/numpy/numpy/pull/30710
        def _get_gamma(virtual_indexes, previous_indexes, method):
            gamma = numpy.asanyarray(virtual_indexes - previous_indexes)
            gamma = method["fix_gamma"](gamma, virtual_indexes)
            return numpy.asanyarray(gamma, dtype=virtual_indexes.dtype)

        monkeypatch.setattr(
            numpy.lib._function_base_impl, "_get_gamma", _get_gamma
        )

    yield


def for_all_methods(name="method"):
    return pytest.mark.parametrize(name, _all_methods)


def _make_nan_array(xp, shape, dtype, period=3):
    # Random array in which every ``period``-th element (in C order) is NaN.
    # A fixed pattern keeps the tests reproducible and guarantees that no
    # slice along the last axis is entirely NaN for the shapes used here.
    a = testing.shaped_random(shape, xp, dtype)
    if numpy.dtype(dtype).char in "efdFD":
        mask = testing.shaped_arange(shape, xp, "int32") % period == 0
        a = xp.where(mask, xp.array(numpy.nan, dtype=dtype), a)
    return a


def test_percentile_kernel_accepts_large_dimensions():
    indices = cupy.array([0], dtype=cupy.float64)
    a = cupy.broadcast_to(cupy.array([1], dtype=cupy.float64), (2**31,))
    out = cupy.empty(1, dtype=cupy.float64)
    order_module._get_percentile_weightnening_kernel()(
        indices, a, 0, a.size, out
    )
    assert out[0] == 1


@testing.with_requires("numpy>=1.22.0rc1")
class TestQuantile:
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_percentile_unexpected_method(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
            with pytest.raises(ValueError):
                xp.percentile(a, q, axis=-1, method="deadbeef")

    # See gh-4453
    @testing.for_float_dtypes()
    @pytest.mark.thread_unsafe(reason="allocator setting not thread-safe")
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
            percentiles = cupy.percentile(a, q, axis=None, method="linear")
        finally:
            cuda.set_allocator(original_allocator)

        assert not cupy.any(cupy.isnan(percentiles))

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_unexpected_method(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.quantile(a, q, axis=-1, method="deadbeef")


@pytest.mark.usefixtures("_fix_gamma")
@testing.with_requires("numpy>=2.0")
@for_all_methods()
class TestQuantileMethods:
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_defaults(self, xp, dtype, method):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = testing.shaped_random((3,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_q_list(self, xp, dtype, method):
        a = testing.shaped_arange((1001,), xp, dtype)
        q = [99, 99.9]
        return xp.percentile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_no_axis(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 4, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=None, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_neg_axis(self, xp, dtype, method):
        a = testing.shaped_random((4, 3, 10, 2, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_tuple_axis(self, xp, dtype, method):
        a = testing.shaped_random((1, 6, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=(0, 1, 2), method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_scalar_q(self, xp, dtype, method):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = 13.37
        return xp.percentile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_percentile_keepdims(self, xp, dtype, method):
        a = testing.shaped_random((7, 2, 9, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.percentile(a, q, axis=None, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)  # NumPy raises error on int8
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_out(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.percentile(a, q, axis=-1, method=method, out=out)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_percentile_overwrite(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        ap = a.copy()
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        res = xp.percentile(
            ap, q, axis=-1, method=method, overwrite_input=True
        )

        assert not xp.all(ap == a)
        return res

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_percentile_bad_q(self, dtype, method):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=100)
            with pytest.raises(ValueError):
                xp.percentile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_percentile_out_of_range_q(self, dtype, method):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            for q in [[-0.1], [100.1]]:
                with pytest.raises(ValueError):
                    xp.percentile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_defaults(self, xp, dtype, method):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = testing.shaped_random((3,), xp, scale=1)
        return xp.quantile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_quantile_q_list(self, xp, dtype, method):
        a = testing.shaped_arange((1001,), xp, dtype)
        q = [0.99, 0.999]
        return xp.quantile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_quantile_no_axis(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 4, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=None, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_neg_axis(self, xp, dtype, method):
        a = testing.shaped_random((4, 3, 10, 2, 8), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_tuple_axis(self, xp, dtype, method):
        a = testing.shaped_random((1, 6, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=(0, 1, 2), method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_quantile_scalar_q(self, xp, dtype, method):
        a = testing.shaped_random((2, 3, 8), xp, dtype)
        q = 0.1337
        return xp.quantile(a, q, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_quantile_keepdims(self, xp, dtype, method):
        a = testing.shaped_random((7, 2, 9, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=None, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)  # NumPy raises error on int8
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_out(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.quantile(a, q, axis=-1, method=method, out=out)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_overwrite(self, xp, dtype, method):
        a = testing.shaped_random((10, 2, 3, 2), xp, dtype)
        ap = a.copy()
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)

        res = xp.quantile(a, q, axis=-1, method=method, overwrite_input=True)

        assert not xp.all(ap == a)
        return res

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_bad_q(self, dtype, method):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.quantile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_quantile_out_of_range_q(self, dtype, method):
        for xp in (numpy, cupy):
            a = testing.shaped_random((4, 2, 3, 2), xp, dtype)
            for q in [[-0.1], [1.1]]:
                with pytest.raises(ValueError):
                    xp.quantile(a, q, axis=-1, method=method)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_quantile_axis_and_keepdims(self, xp, dtype, method):
        a = testing.shaped_random((1, 6, 3, 2), xp, dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.quantile(a, q, axis=0, keepdims=True, method=method)


# ``contiguous_check=False``: NumPy's NaN code path builds its result from
# fancy-indexed intermediates, which come out F-ordered, while CuPy returns
# the C-ordered result. Only the memory layout differs, so the flag
# comparison is switched off for these tests.
@pytest.mark.usefixtures("_fix_gamma")
@testing.with_requires("numpy>=2.0")
@for_all_methods()
class TestNaNQuantileMethods:
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_defaults(self, xp, dtype, method):
        a = _make_nan_array(xp, (2, 3, 8), dtype)
        q = testing.shaped_random((3,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_q_list(self, xp, dtype, method):
        a = _make_nan_array(xp, (1001,), dtype)
        q = [99, 99.9]
        return xp.nanpercentile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_no_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 4, 8), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, axis=None, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_neg_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (4, 3, 10, 2, 8), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, axis=-1, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_tuple_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (1, 6, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, axis=(0, 1, 2), method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_scalar_q(self, xp, dtype, method):
        a = _make_nan_array(xp, (2, 3, 8), dtype)
        q = 13.37
        return xp.nanpercentile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, contiguous_check=False)
    def test_nanpercentile_keepdims(self, xp, dtype, method):
        a = _make_nan_array(xp, (7, 2, 9, 2), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, axis=None, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, contiguous_check=False)
    def test_nanpercentile_axis_and_keepdims(self, xp, dtype, method):
        a = _make_nan_array(xp, (1, 6, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        return xp.nanpercentile(a, q, axis=0, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_out(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.nanpercentile(a, q, axis=-1, out=out, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanpercentile_overwrite(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 3, 2), dtype)
        ap = a.copy()
        q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
        res = xp.nanpercentile(
            ap, q, axis=-1, method=method, overwrite_input=True
        )

        assert not xp.all(ap == a)
        return res

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_defaults(self, xp, dtype, method):
        a = _make_nan_array(xp, (2, 3, 8), dtype)
        q = testing.shaped_random((3,), xp, scale=1)
        return xp.nanquantile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_q_list(self, xp, dtype, method):
        a = _make_nan_array(xp, (1001,), dtype)
        q = [0.99, 0.999]
        return xp.nanquantile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_no_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 4, 8), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.nanquantile(a, q, axis=None, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_neg_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (4, 3, 10, 2, 8), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.nanquantile(a, q, axis=-1, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_tuple_axis(self, xp, dtype, method):
        a = _make_nan_array(xp, (1, 6, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.nanquantile(a, q, axis=(0, 1, 2), method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_scalar_q(self, xp, dtype, method):
        a = _make_nan_array(xp, (2, 3, 8), dtype)
        q = 0.1337
        return xp.nanquantile(a, q, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, contiguous_check=False)
    def test_nanquantile_keepdims(self, xp, dtype, method):
        a = _make_nan_array(xp, (7, 2, 9, 2), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.nanquantile(a, q, axis=None, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5, contiguous_check=False)
    def test_nanquantile_axis_and_keepdims(self, xp, dtype, method):
        a = _make_nan_array(xp, (1, 6, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        return xp.nanquantile(a, q, axis=0, keepdims=True, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_out(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 3, 2), dtype)
        q = testing.shaped_random((5,), xp, scale=1)
        out = testing.shaped_random((5, 10, 2, 3), xp, dtype)
        return xp.nanquantile(a, q, axis=-1, out=out, method=method)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, contiguous_check=False)
    def test_nanquantile_overwrite(self, xp, dtype, method):
        a = _make_nan_array(xp, (10, 2, 3, 2), dtype)
        ap = a.copy()
        q = testing.shaped_random((5,), xp, scale=1)
        res = xp.nanquantile(
            ap, q, axis=-1, method=method, overwrite_input=True
        )

        assert not xp.all(ap == a)
        return res


@testing.with_requires("numpy>=1.22.0rc1")
class TestNaNQuantile:
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_nanquantile_bad_q(self, dtype):
        for xp in (numpy, cupy):
            a = _make_nan_array(xp, (4, 2, 3, 2), xp.float64)
            q = testing.shaped_random((1, 2, 3), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.nanquantile(a, q, axis=-1)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_nanquantile_out_of_range_q(self, dtype):
        for xp in (numpy, cupy):
            a = _make_nan_array(xp, (4, 2, 3, 2), xp.float64)
            for q in [[-0.1], [1.1]]:
                with pytest.raises(ValueError):
                    xp.nanquantile(a, q, axis=-1)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_nanpercentile_unexpected_method(self, dtype):
        for xp in (numpy, cupy):
            a = _make_nan_array(xp, (4, 2, 3, 2), xp.float64)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=100)
            with pytest.raises(ValueError):
                xp.nanpercentile(a, q, axis=-1, method="deadbeef")

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_nanquantile_unexpected_method(self, dtype):
        for xp in (numpy, cupy):
            a = _make_nan_array(xp, (4, 2, 3, 2), xp.float64)
            q = testing.shaped_random((5,), xp, dtype=dtype, scale=1)
            with pytest.raises(ValueError):
                xp.nanquantile(a, q, axis=-1, method="deadbeef")

    @for_all_methods()
    def test_nanquantile_without_nan_equals_quantile(self, method):
        # NaNs cannot change the result, so every method must agree with the
        # non-NaN code path, including the dtype in which the interpolation
        # difference is computed (float32 ``a`` with a float64 ``q``).
        # The two paths are not bit-identical: nvcc contracts the kernel's
        # ``a + diff * weight`` into an fma, while this path evaluates it with
        # separate array ops, which can differ by one ulp.
        for dtype, rtol in ((cupy.float64, 1e-14), (cupy.float32, 1e-7)):
            a = testing.shaped_random((4, 7), cupy, dtype)
            testing.assert_allclose(
                cupy.nanquantile(a, 0.3, axis=1, method=method),
                cupy.quantile(a, 0.3, axis=1, method=method),
                rtol=rtol,
            )

    @for_all_methods()
    def test_nanpercentile_without_nan_equals_percentile(self, method):
        for dtype, rtol in ((cupy.float64, 1e-14), (cupy.float32, 1e-7)):
            a = testing.shaped_random((4, 7), cupy, dtype)
            testing.assert_allclose(
                cupy.nanpercentile(a, 30, axis=1, method=method),
                cupy.percentile(a, 30, axis=1, method=method),
                rtol=rtol,
            )

    @for_all_methods()
    def test_nanquantile_integer_index_stays_gather(self, method):
        # Regression test: the NaN path must not promote the integer indices
        # of the integer methods, otherwise ``inverted_cdf`` interpolates
        # instead of gathering and turns inf into NaN.
        a = cupy.array([[1.0, numpy.inf], [2.0, 3.0]])
        res = cupy.nanquantile(a, 1.0, axis=1, method=method)
        testing.assert_array_equal(
            res,
            cupy.asnumpy(
                numpy.nanquantile(cupy.asnumpy(a), 1.0, axis=1, method=method)
            ),
        )

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_nanquantile_integer_input(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        return xp.nanquantile(a, [0.25, 0.75], axis=1)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_nanpercentile_integer_input(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        return xp.nanpercentile(a, [25, 75], axis=1)

    @testing.for_complex_dtypes()
    def test_nanquantile_complex_input(self, dtype):
        # NumPy rejects complex input for the NaN-aware variants.
        for xp in (numpy, cupy):
            a = testing.shaped_random((3, 4), xp, dtype)
            with pytest.raises(TypeError):
                xp.nanquantile(a, 0.5, axis=1)

    @testing.for_complex_dtypes()
    def test_nanpercentile_complex_input(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((3, 4), xp, dtype)
            with pytest.raises(TypeError):
                xp.nanpercentile(a, 50, axis=1)

    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_nanquantile_single_element_axis(self, xp):
        a = xp.array([[numpy.nan, 2.0], [3.0, numpy.nan]])
        return xp.nanquantile(a, 0.5, axis=1)

    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_nanquantile_all_single_element_slices(self, xp):
        # Every slice along ``axis=1`` holds exactly one element.
        a = xp.array([[1.0], [numpy.nan], [3.0]])
        return xp.nanquantile(a, [0.0, 1.0], axis=1)

    @testing.numpy_cupy_allclose()
    def test_nanquantile_empty_axis(self, xp):
        # Every slice is empty. NumPy short-circuits empty input to
        # ``nanmean`` (see ``_nanquantile_unchecked``), so ``q`` is dropped
        # from the result shape; that quirk is reproduced here.
        a = xp.zeros((2, 0))
        return xp.nanquantile(a, 0.5, axis=1)

    @testing.numpy_cupy_allclose()
    def test_nanquantile_empty_axis_sequence_q(self, xp):
        a = xp.zeros((2, 0))
        return xp.nanquantile(a, [0.25, 0.75], axis=1)

    @testing.numpy_cupy_allclose()
    def test_nanquantile_empty_kept_axis(self, xp):
        a = xp.zeros((0, 3))
        return xp.nanquantile(a, [0.25, 0.75], axis=1)

    @testing.numpy_cupy_allclose()
    def test_nanquantile_empty_input(self, xp):
        a = xp.zeros((0,))
        return xp.nanquantile(a, 0.5)

    @testing.numpy_cupy_allclose()
    def test_nanpercentile_empty_axis(self, xp):
        a = xp.zeros((2, 0))
        return xp.nanpercentile(a, 50, axis=1)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_nanquantile_empty_axis_keeps_input_dtype(self, xp, dtype):
        # The empty short circuit mirrors ``numpy.nanmean``, which keeps the
        # inexact dtype of its input instead of promoting with ``q``.
        a = xp.zeros((2, 0), dtype=dtype)
        return xp.nanquantile(a, [0.25, 0.75], axis=1)

    def test_nanquantile_integer_q(self):
        # Regression test: the per-slice valid count must not be stored in
        # the dtype of ``q``. 300 valid elements do not fit in int8, so the
        # count would wrap negative and every index would clamp to 0.
        # NumPy cannot be compared against here: it raises OverflowError on
        # the same input (``(n - 1) * quantiles`` in int8).
        a = _make_nan_array(cupy, (5, 300), cupy.float64)
        got = cupy.nanquantile(a, cupy.array([0, 1], dtype=cupy.int8), axis=1)
        testing.assert_allclose(
            got, cupy.nanquantile(a, [0.0, 1.0], axis=1), rtol=1e-14
        )
        # q=1 must not collapse onto q=0, which is what a wrapped, negative
        # valid count would give.
        assert (got[1] != got[0]).all()

    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_nanquantile_scalar_and_sequence_q(self, xp):
        a = _make_nan_array(xp, (5, 3), xp.float64)
        scalar = xp.nanquantile(a, 0.375, axis=1)
        seq = xp.nanquantile(a, [0.375, 0.75], axis=1)
        assert scalar.shape == (5,)
        assert seq.shape == (2, 5)
        # A scalar q must give exactly the matching entry of a sequence q.
        testing.assert_allclose(scalar, seq[0], rtol=1e-7)
        return seq

    @testing.numpy_cupy_allclose()
    def test_nanquantile_fractional_q(self, xp):
        a = _make_nan_array(xp, (11,), xp.float64)
        return xp.nanquantile(a, [1 / 3, 2 / 3, 1 / 7, 5 / 7])

    @testing.numpy_cupy_allclose()
    def test_nanquantile_1d_out(self, xp):
        a = _make_nan_array(xp, (9,), xp.float64)
        out = testing.shaped_random((4,), xp, xp.float64)
        return xp.nanquantile(a, [0.1, 0.4, 0.5, 0.9], out=out)

    @testing.numpy_cupy_allclose()
    def test_nanquantile_nd_out(self, xp):
        a = _make_nan_array(xp, (4, 3, 2), xp.float64)
        out = testing.shaped_random((3, 4, 2), xp, xp.float64)
        return xp.nanquantile(a, [0.2, 0.5, 0.8], axis=1, out=out)

    @pytest.mark.parametrize(
        "axis, data",
        [
            (0, [[numpy.nan, 1.0], [numpy.nan, 2.0]]),
            (1, [[numpy.nan, numpy.nan], [1.0, 2.0]]),
            (-1, [[numpy.nan, numpy.nan], [1.0, 2.0]]),
        ],
    )
    def test_all_nan_slice_warns_and_returns_nan(self, axis, data):
        a = cupy.array(data)
        # ``axis=None`` and ``axis=(0, 1)`` are left out on purpose: there the
        # single remaining slice is the whole array, which is not all NaN.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = cupy.nanquantile(a, 0.5, axis=axis)
        assert any(
            issubclass(w.category, RuntimeWarning)
            and "All-NaN slice encountered" in str(w.message)
            for w in caught
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = numpy.nanquantile(cupy.asnumpy(a), 0.5, axis=axis)
        testing.assert_allclose(res, expected, rtol=1e-7)

    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
    def test_partial_nan_slice_does_not_warn(self, axis):
        a = _make_nan_array(cupy, (4, 5), cupy.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cupy.nanquantile(a, 0.5, axis=axis)
        assert not [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]

    def test_out_prefilled_with_nan_does_not_warn(self):
        # The all-NaN check must not look at a user-supplied ``out`` buffer.
        # A sequence q is used because scalar q with out= is unsupported by
        # ``cupy.quantile`` as well (pre-existing, unrelated to NaNs).
        a = cupy.array([[1.0, 2.0], [3.0, numpy.nan]])
        out = cupy.full((1, 2), numpy.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = cupy.nanquantile(a, [0.5], axis=1, out=out)
        assert not [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        testing.assert_allclose(res, [[1.5, 3.0]])


class TestOrder:
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
        a = xp.array([float("nan"), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmax(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_all_nan(self, xp, dtype):
        a = xp.array([float("nan"), float("nan")], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
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
        a = xp.array([float("nan"), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmin(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_all_nan(self, xp, dtype):
        a = xp.array([float("nan"), float("nan")], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
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
        a = xp.array([float("nan"), 1, -1], dtype)
        return xp.ptp(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_all_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float("nan"), float("nan")], dtype)
        return xp.ptp(a)


# See gh-4607
# "Magic" values used in this test were empirically found to result in
# non-monotonicity for less accurate linear interpolation formulas
@testing.parameterize(
    *testing.product(
        {
            "magic_value": (
                -29,
                -53,
                -207,
                -16373,
                -99999,
            )
        }
    )
)
class TestPercentileMonotonic:
    @testing.with_requires("numpy>=1.22.0rc1")
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_percentile_monotonic(self, dtype, xp):
        a = testing.shaped_random((5,), xp, dtype)

        a[0] = self.magic_value
        a[1] = self.magic_value
        q = xp.linspace(0, 100, 21)
        percentiles = xp.percentile(a, q, method="linear")

        # Assert that percentile output increases monotonically
        assert xp.all(xp.diff(percentiles) >= 0)

        return percentiles
