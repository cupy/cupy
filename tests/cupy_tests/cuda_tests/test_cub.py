import functools
import unittest

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
    t_cub = 0.
    for i in range(runs):
        test.start.record()
        getattr(arr, func)(axis=axis)
        test.stop.record()
        test.stop.synchronize()
        t_cub += cupy.cuda.get_elapsed_time(test.start, test.stop)

    cupy.cuda.cub_enabled = False
    t_cupy = 0.
    for i in range(runs):
        test.start.record()
        getattr(arr, func)(axis=axis)
        test.stop.record()
        test.stop.synchronize()
        t_cupy += cupy.cuda.get_elapsed_time(test.start, test.stop)

    # TODO(leofang): raise PerformanceWarning here?
    print("CUB: {:10.5f}; CuPy: {:10.5f} (ms), for {} runs, shape={}, axis={},"
          " dtype={}".format(t_cub, t_cupy, runs, arr.shape, axis, arr.dtype))
    cupy.cuda.cub_enabled = True  # restore


# This class compares CUB results against NumPy's
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
}))
@testing.gpu
class TestCUBreduction(unittest.TestCase):
    @for_contiguous_axes()
    @testing.for_dtypes('lLfdFD')  # sum supports less dtypes
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_sum(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.sum(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('bhilBHILfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_min(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.min(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('bhilBHILfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_max(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.max(axis=axis)

    # argmin does not support axis yet
    @testing.for_dtypes('bhilBHILfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_argmin(self, xp, dtype):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.argmin()

    # argmax does not support axis yet
    @testing.for_dtypes('bhilBHILfdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5)
    def test_cub_argmax(self, xp, dtype):
        assert cupy.cuda.cub_enabled
        a = testing.shaped_random(self.shape, xp, dtype)
        return a.argmax()


# This class compares CUB results against CuPy's original implementation
# and their performance
@testing.parameterize(*testing.product({
    'shape': [(1024,), (256, 1024), (128, 256, 256), (4, 128, 256, 256)],
}))
@testing.slow
@testing.gpu
class TestCUBperformance(unittest.TestCase):
    def setUp(self):
        self.start = cupy.cuda.Event()
        self.stop = cupy.cuda.Event()

    def tearDown(self):
        del self.start, self.stop
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @for_contiguous_axes()
    @testing.for_dtypes('lLfdFD')  # sum supports less dtypes
    def test_cub_sum_performance(self, dtype, axis):
        a = testing.shaped_random(self.shape, cupy, dtype)
        timing(self, 'sum', a, axis)

    @for_contiguous_axes()
    @testing.for_dtypes('bhilBHILfdFD')
    def test_cub_min_performance(self, dtype, axis):
        a = testing.shaped_random(self.shape, cupy, dtype)
        timing(self, 'min', a, axis)

    @for_contiguous_axes()
    @testing.for_dtypes('bhilBHILfdFD')
    def test_cub_max_performance(self, dtype, axis):
        a = testing.shaped_random(self.shape, cupy, dtype)
        timing(self, 'max', a, axis)

    # argmin does not support axis yet
    @testing.for_dtypes('bhilBHILfdFD')
    def test_cub_argmin_performance(self, dtype):
        a = testing.shaped_random(self.shape, cupy, dtype)
        timing(self, 'argmin', a, None)

    # argmax does not support axis yet
    @testing.for_dtypes('bhilBHILfdFD')
    def test_cub_argmax_performance(self, dtype):
        a = testing.shaped_random(self.shape, cupy, dtype)
        timing(self, 'argmax', a, None)
