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
    """Decorator for parameterized tests with and wihout CUB support.
    Tests are repeated with cupy.cuda.cub_enabled set to True and False

    Args:
        axis(str): Argument name to which specified axis are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.

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
                    print(name, 'is', a, 'ndim is', ndim, 'shape is', self.shape)
                    raise
        return test_func
    return decorator


#def enable_CUB(flags=[True, False], name='enable_cub'):
#    """Decorator for parameterized tests with and wihout CUB support.
#    Tests are repeated with cupy.cuda.cub_enabled set to True and False
#
#    Args:
#         flags(list of bool): The boolean cases to test.
#         name(str): Argument name to which specified dtypes are passed.
#
#    This decorator adds a keyword argument specified by ``name``
#    to the test fixture. Then, it runs the fixtures in parallel
#    by passing the each element of ``dtypes`` to the named
#    argument.
#
#    Adapted from tests/cupy_tests/fft_tests/test_fft.py.
#    """
#    def decorator(impl):
#        @functools.wraps(impl)
#        def test_func(self, *args, **kw):
#            # get original flag
#            flag = cupy.cuda.cub_enabled
#            try:
#                for f in flags:
#                    try:
#                        # enable or disable CUB
#                        cupy.cuda.cub_enabled = f
#
#                        kw[name] = f
#                        impl(self, *args, **kw)
#                    except Exception:
#                        print(name, 'is', f)
#                        raise
#            finally:
#                # restore original global flag
#                cupy.cuda.cub_enabled = flag
#        return test_func
#    return decorator


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
#    if t_cub > 1.05 * t_cupy:
#        warnings.warn("CUB: "+str(t_cub)+"; CuPy: "+str(t_cupy)+" (shape={}, axis={}, dtype={})".format(arr.shape, axis, arr.dtype), cupy.util.PerformanceWarning)
    print("CUB:", t_cub, '; CuPy:', t_cupy, '(ms for', runs, 'runs)')

    cupy.cuda.cub_enabled = True  # restore


#axis_cases = []
#for r in range(1, 5):
#    data = list(permutations((0, 1, 2, 3), r))
#    axis_cases += list(set(tuple(sorted(item)) for item in data))


# modified from tests/cupy_tests/math_tests/test_sumprod.py
@testing.parameterize(*testing.product({
#    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
    'shape': [(256,), (256, 256), (128, 256, 256), (4, 128, 256, 256)],
#    'axis': axis_cases,
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
    def test_sum(self, xp, dtype, axis):
        assert cupy.cuda.cub_enabled
        #print("testing: xp={}, dtype={}, shape={}, axis={}".format(xp, dtype, self.shape, axis))
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.sum(axis=axis)

    @for_contiguous_axes()
    @testing.for_dtypes('LQfdFD')
    def test_sum_cub_performance(self, dtype, axis):
        print("testing: dtype={}, shape={}, axis={}".format(dtype, self.shape, axis))
        a = testing.shaped_arange(self.shape, cupy, dtype)
        timing(self, 'sum', a, axis)


#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_all(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype)
#        return a.sum()
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_external_sum_all(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype)
#        return xp.sum(a)
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_all2(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40), xp, dtype)
#        return a.sum()
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_all_transposed(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
#        return a.sum()
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_all_transposed2(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
#        return a.sum()
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_axis(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype)
#        return a.sum(axis=1)
#
#    @testing.slow
#    @testing.with_requires('numpy>=1.10')
#    @testing.numpy_cupy_allclose()
#    def test_sum_axis_huge(self, xp):
#        a = testing.shaped_random((2048, 1, 1024), xp, 'b')
#        a = xp.broadcast_to(a, (2048, 1024, 1024))
#        return a.sum(axis=2)
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_external_sum_axis(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype)
#        return xp.sum(a, axis=1)
#
#    # float16 is omitted, since NumPy's sum on float16 arrays has more error
#    # than CuPy's.
#    @testing.for_all_dtypes(no_float16=True)
#    @testing.numpy_cupy_allclose()
#    def test_sum_axis2(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40), xp, dtype)
#        return a.sum(axis=1)
#
#    def test_sum_axis2_float16(self):
#        # Note that the above test example overflows in float16. We use a
#        # smaller array instead.
#        a = testing.shaped_arange((2, 30, 4), dtype='e')
#        sa = a.sum(axis=1)
#        b = testing.shaped_arange((2, 30, 4), numpy, dtype='f')
#        sb = b.sum(axis=1)
#        testing.assert_allclose(sa, sb.astype('e'))
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose(contiguous_check=False)
#    def test_sum_axis_transposed(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
#        return a.sum(axis=1)
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose(contiguous_check=False)
#    def test_sum_axis_transposed2(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
#        return a.sum(axis=1)
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_axes(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
#        return a.sum(axis=(1, 3))
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose(rtol=1e-4)
#    def test_sum_axes2(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
#        return a.sum(axis=(1, 3))
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose(rtol=1e-6)
#    def test_sum_axes3(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
#        return a.sum(axis=(0, 2, 3))
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose(rtol=1e-6)
#    def test_sum_axes4(self, xp, dtype):
#        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
#        return a.sum(axis=(0, 2, 3))
#
#    @testing.for_all_dtypes_combination(names=['src_dtype', 'dst_dtype'])
#    @testing.numpy_cupy_allclose()
#    def test_sum_dtype(self, xp, src_dtype, dst_dtype):
#        if not xp.can_cast(src_dtype, dst_dtype):
#            return xp.array([])  # skip
#        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
#        return a.sum(dtype=dst_dtype)
#
#    @testing.numpy_cupy_allclose()
#    def test_sum_keepdims(self, xp):
#        a = testing.shaped_arange((2, 3, 4), xp)
#        return a.sum(axis=1, keepdims=True)
#
#    @testing.for_all_dtypes()
#    @testing.numpy_cupy_allclose()
#    def test_sum_out(self, xp, dtype):
#        a = testing.shaped_arange((2, 3, 4), xp, dtype)
#        b = xp.empty((2, 4), dtype=dtype)
#        a.sum(axis=1, out=b)
#        return b
#
#    def test_sum_out_wrong_shape(self):
#        a = testing.shaped_arange((2, 3, 4))
#        b = cupy.empty((2, 3))
#        with self.assertRaises(ValueError):
#            a.sum(axis=1, out=b)
