from contextlib import contextmanager
import functools
from itertools import permutations
import pytest
import unittest
import warnings

import numpy

import cupy
from cupy import testing


if cupy.cuda.cub_enabled is False:
    raise unittest.SkipTest("The CUB module is not built. Skip the tests.")


def for_contiguous_axes(name='axis'):
    """Decorator for parametrizing tests with possible contiguous axes.

    Args:
        axis(str): Argument name to which specified axis are passed.

    Adapted from tests/cupy_tests/fft_tests/test_fft.py.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            ndim = len(self.shape)
            for i in range(ndim):
                a = ()
                for j in range(ndim-1, i-1, -1):
                    a = (j,) + a
                try:
                    kw[name] = a
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', a, ', ndim is', ndim, ', shape is',
                          self.shape)
                    raise
        return test_func
    return decorator


def timing(test, func, arr, axis, runs=20):
    cupy.cuda.cub_enabled = True
    test.start.record()
    for i in range(runs):
        getattr(arr, func)(axis=axis)
    test.stop.record()
    test.stop.synchronize()
    t_cub = cupy.cuda.get_elapsed_time(test.start, test.stop)

    cupy.cuda.cub_enabled = False
    test.start.record()
    for i in range(runs):
        getattr(arr, func)(axis=axis)
    test.stop.record()
    test.stop.synchronize()
    t_cupy = cupy.cuda.get_elapsed_time(test.start, test.stop)

    # TODO(leofang): raise PerformanceWarning here?
    print("CUB: {:4.5f}; CuPy: {:4.5f} (ms for {} runs, shape={}, axis={})"
          .format(t_cub, t_cupy, runs, arr.shape, axis))
    cupy.cuda.cub_enabled = True  # restore


# Tests named with test_cub_{op} compare CUB results against NumPy's,
# and those named with test_cub_{op}_performance compare CUB results
# against CuPy's original implementation while timing.
@testing.parameterize(*testing.product({
#    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
    'shape': [(256,), (256, 256), (128, 256, 256), (4, 128, 256, 256)],
}))
@testing.gpu
class TestCUBreduction(unittest.TestCase):
    def setUp(self):
        self.start = cupy.cuda.Event()
        self.stop = cupy.cuda.Event()

    def tearDown(self):
        del self.start, self.stop
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_sum(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.sum(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    def test_cub_sum_performance(self, dtype, axis):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'sum', a, axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_min(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.min(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    def test_cub_min_performance(self, dtype, axis):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'min', a, axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_max(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.max(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    def test_cub_max_performance(self, dtype, axis):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'max', a, axis)

    # argmin does not support axis yet
    @testing.for_dtypes('LQfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_argmin(self, xp, dtype):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.argmin()

    # argmin does not support axis yet
    @testing.for_dtypes('LQfdFD')
    def test_cub_argmin_performance(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'argmin', a, None)

    # argmax does not support axis yet
    @testing.for_dtypes('LQfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_argmax(self, xp, dtype):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.argmax()

    # argmax does not support axis yet
    @testing.for_dtypes('LQfdFD')
    def test_cub_argmax_performance(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'argmax', a, None)
