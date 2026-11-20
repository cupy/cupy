from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing


# Reductions with few outputs and a long reduction axis, e.g. the column
# reduction of a tall C-ordered matrix, are split over several blocks per
# output group (two kernels). These shapes take that path; the small ones
# do not and act as controls.
_shapes_axes = [
    ((200003, 3), 0),
    ((100001, 64), 0),
    ((3, 40001, 5), (0, 1)),
    ((40001, 4, 3), 0),
    ((8, 30001, 2), 1),
    ((70001, 1), 0),
    ((20000, 200), 0),
    ((50, 7), 0),
]


@pytest.mark.parametrize('shape,axis', _shapes_axes)
class TestSplitReduction:

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-2, contiguous_check=False)
    def test_sum(self, xp, dtype, shape, axis):
        a = testing.shaped_random(shape, xp, dtype, scale=10)
        return a.sum(axis=axis)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-4)
    def test_mean_var_std(self, xp, dtype, shape, axis):
        a = testing.shaped_random(shape, xp, dtype)
        return xp.stack([a.mean(axis=axis), a.var(axis=axis),
                         a.std(axis=axis, ddof=1)])

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_max_min(self, xp, dtype, shape, axis):
        a = testing.shaped_random(shape, xp, dtype, scale=100)
        return xp.stack([a.max(axis=axis), a.min(axis=axis)])

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_argmax_argmin(self, xp, dtype, shape, axis):
        # The reduction index _J must stay global across the split.
        if isinstance(axis, tuple):
            pytest.skip('argmax takes a single axis')
        a = testing.shaped_random(shape, xp, dtype, scale=1000)
        return xp.stack([a.argmax(axis=axis), a.argmin(axis=axis)])

    @testing.numpy_cupy_array_equal()
    def test_any_all_count(self, xp, shape, axis):
        a = testing.shaped_random(shape, xp, numpy.float32) > 0.7
        return xp.stack([a.any(axis=axis), a.all(axis=axis),
                         xp.count_nonzero(a, axis=axis)])

    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-4)
    def test_nan_reductions(self, xp, shape, axis):
        a = testing.shaped_random(shape, xp, numpy.float64)
        a.ravel()[::11] = xp.nan
        return xp.stack([xp.nansum(a, axis=axis), xp.nanmean(a, axis=axis),
                         xp.nanvar(a, axis=axis), xp.nanmax(a, axis=axis)])

    # contiguous_check: CuPy returns C-ordered keepdims results for F inputs
    # (unrelated to the split path, same on main).
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-3, contiguous_check=False)
    def test_f_order_keepdims(self, xp, shape, axis):
        a = xp.asfortranarray(testing.shaped_random(shape, xp, numpy.float64))
        return a.sum(axis=axis, keepdims=True)

    def test_out(self, shape, axis):
        a = testing.shaped_random(shape, cupy, numpy.float32)
        out = cupy.empty(a.sum(axis=axis).shape, numpy.float64)
        cupy.sum(a, axis=axis, dtype=numpy.float64, out=out)
        testing.assert_allclose(
            out, a.get().astype(numpy.float64).sum(axis=axis), rtol=1e-6)

    def test_reduction_kernel(self, shape, axis):
        l2 = cupy.ReductionKernel(
            'T x', 'T y', 'x * x', 'a + b', 'y = sqrt(a)', '0', 'split_l2')
        # Uses the reduction index _J: count of elements at even positions
        # along the (flattened) reduction axes.
        even = cupy.ReductionKernel(
            'T x', 'int64 y', '(_J % 2 == 0) ? 1 : 0', 'a + b', 'y = a',
            '0', 'split_even')
        a = testing.shaped_random(shape, cupy, numpy.float64)
        h = a.get()
        testing.assert_allclose(
            l2(a, axis=axis), numpy.sqrt((h * h).sum(axis=axis)), rtol=1e-6)
        axes = axis if isinstance(axis, tuple) else (axis,)
        n = int(numpy.prod([shape[i] for i in axes]))
        testing.assert_array_equal(
            even(a, axis=axis), numpy.full(l2(a, axis=axis).shape,
                                           (n + 1) // 2))
