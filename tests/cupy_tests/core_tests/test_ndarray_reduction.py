import unittest

import numpy
import pytest

import cupy
import cupy._core._accelerator as _acc
from cupy._core import _cub_reduction
from cupy import testing


@testing.parameterize(*testing.product({
    'order': ('C', 'F'),
}))
@testing.gpu
class TestArrayReduction(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.max(keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype, order=self.order)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.max(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.max(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.max(axis=(1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.max(axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_max_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype, order=self.order)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.min()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.min(keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype, order=self.order)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.min(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.min(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.min(axis=(1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.min(axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_min_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype, order=self.order)
        return a.min()

    # skip bool: numpy's ptp raises a TypeError on bool inputs
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.ptp()

    @testing.with_requires('numpy>=1.15')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.ptp(keepdims=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype, order=self.order)
        return a.ptp(axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.ptp(axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.ptp(axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.ptp(axis=2)

    @testing.with_requires('numpy>=1.15')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.ptp(axis=(1, 2))

    @testing.with_requires('numpy>=1.15')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.ptp(axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.ptp()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.ptp()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ptp_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype, order=self.order)
        return a.ptp()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype, order=self.order)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmax(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmax(axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.argmax()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.argmax()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmax_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype, order=self.order)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, order=self.order)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype, order=self.order)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmin(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype, order=self.order)
        return a.argmin(axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_nan(self, xp, dtype):
        if _acc.ACCELERATOR_CUTENSOR in _acc.get_routine_accelerators():
            pytest.skip()
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.argmin()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_nan_real(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype, order=self.order)
        return a.argmin()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_argmin_nan_imag(self, xp, dtype):
        a = xp.array([float('nan')*1.j, 1.j, -1.j], dtype, order=self.order)
        return a.argmin()


@testing.parameterize(*testing.product({
    # TODO(leofang): make a @testing.for_all_axes decorator
    'shape_and_axis': [
        ((), None),
        ((0,), (0,)),
        ((0, 2), (0,)),
        ((0, 2), (1,)),
        ((0, 2), (0, 1)),
        ((2, 0), (0,)),
        ((2, 0), (1,)),
        ((2, 0), (0, 1)),
        ((0, 2, 3), (0,)),
        ((0, 2, 3), (1,)),
        ((0, 2, 3), (2,)),
        ((0, 2, 3), (0, 1)),
        ((0, 2, 3), (1, 2)),
        ((0, 2, 3), (0, 2)),
        ((0, 2, 3), (0, 1, 2)),
        ((2, 0, 3), (0,)),
        ((2, 0, 3), (1,)),
        ((2, 0, 3), (2,)),
        ((2, 0, 3), (0, 1)),
        ((2, 0, 3), (1, 2)),
        ((2, 0, 3), (0, 2)),
        ((2, 0, 3), (0, 1, 2)),
        ((2, 3, 0), (0,)),
        ((2, 3, 0), (1,)),
        ((2, 3, 0), (2,)),
        ((2, 3, 0), (0, 1)),
        ((2, 3, 0), (1, 2)),
        ((2, 3, 0), (0, 2)),
        ((2, 3, 0), (0, 1, 2)),
    ],
    'order': ('C', 'F'),
    'func': ('min', 'max', 'argmax', 'argmin'),
}))
class TestArrayReductionZeroSize:

    @testing.numpy_cupy_allclose(
        contiguous_check=False, accept_error=ValueError)
    def test_zero_size(self, xp):
        shape, axis = self.shape_and_axis
        # NumPy only supports axis being an int
        if self.func in ('argmax', 'argmin'):
            if axis is not None and len(axis) == 1:
                axis = axis[0]
            else:
                pytest.skip(
                    f"NumPy does not support axis={axis} for {self.func}")
        # dtype is irrelevant here, just pick one
        a = testing.shaped_random(shape, xp, xp.float32, order=self.order)
        return getattr(a, self.func)(axis=axis)


# This class compares CUB results against NumPy's. ("fallback" is CuPy's
# original kernel, also tested here to reduce code duplication.)
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40),
              # skip (2, 3, 0) because it would not hit the CUB code path
              (0,), (2, 0), (0, 2), (0, 2, 3), (2, 3, 0)],
    'order': ('C', 'F'),
    'backend': ('device', 'block', 'fallback'),
}))
@pytest.mark.skipif(
    not cupy.cuda.cub.available, reason='The CUB routine is not enabled')
class TestCubReduction:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.old_routine_accelerators = _acc.get_routine_accelerators()
        self.old_reduction_accelerators = _acc.get_reduction_accelerators()
        if self.backend == 'device':
            _acc.set_routine_accelerators(['cub'])
            _acc.set_reduction_accelerators([])
        elif self.backend == 'block':
            _acc.set_routine_accelerators([])
            _acc.set_reduction_accelerators(['cub'])
        elif self.backend == 'fallback':
            _acc.set_routine_accelerators([])
            _acc.set_reduction_accelerators([])
        yield
        _acc.set_routine_accelerators(self.old_routine_accelerators)
        _acc.set_reduction_accelerators(self.old_reduction_accelerators)

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(
        contiguous_check=False, accept_error=ValueError)
    def test_cub_min(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)

        if xp is numpy:
            return a.min(axis=axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if self.backend == 'device':
            func_name = 'cupy._core._routines_statistics.cub.'
            if len(axis) == len(self.shape):
                func_name += 'device_reduce'
            else:
                func_name += 'device_segmented_reduce'
            with testing.AssertFunctionIsCalled(func_name, return_value=ret):
                a.min(axis=axis)
        elif self.backend == 'block':
            # this is the only function we can mock; the rest is cdef'd
            func_name = 'cupy._core._cub_reduction.'
            func_name += '_SimpleCubReductionKernel_get_cached_function'
            func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
            if len(axis) == len(self.shape):
                times_called = 2  # two passes
            else:
                times_called = 1  # one pass
            if a.size == 0:
                times_called = 0  # _reduction.pyx has an early return path
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.min(axis=axis)
        elif self.backend == 'fallback':
            pass
        # ...then perform the actual computation
        return a.min(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_cub_min_empty_axis(self, xp, dtype, contiguous_check=False):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.min(axis=())

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(
        contiguous_check=False, accept_error=ValueError)
    def test_cub_max(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)

        if xp is numpy:
            return a.max(axis=axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if self.backend == 'device':
            func_name = 'cupy._core._routines_statistics.cub.'
            if len(axis) == len(self.shape):
                func_name += 'device_reduce'
            else:
                func_name += 'device_segmented_reduce'
            with testing.AssertFunctionIsCalled(func_name, return_value=ret):
                a.max(axis=axis)
        elif self.backend == 'block':
            # this is the only function we can mock; the rest is cdef'd
            func_name = 'cupy._core._cub_reduction.'
            func_name += '_SimpleCubReductionKernel_get_cached_function'
            func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
            if len(axis) == len(self.shape):
                times_called = 2  # two passes
            else:
                times_called = 1  # one pass
            if a.size == 0:
                times_called = 0  # _reduction.pyx has an early return path
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.max(axis=axis)
        elif self.backend == 'fallback':
            pass
        # ...then perform the actual computation
        return a.max(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_cub_max_empty_axis(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.max(axis=())
