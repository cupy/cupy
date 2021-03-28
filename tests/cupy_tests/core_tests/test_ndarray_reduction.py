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


# This class compares CUB results against NumPy's
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
    'order': ('C', 'F'),
    'backend': ('device', 'block'),
}))
@testing.gpu
@unittest.skipUnless(cupy.cuda.cub.available, 'The CUB routine is not enabled')
class TestCubReduction(unittest.TestCase):

    def setUp(self):
        self.old_routine_accelerators = _acc.get_routine_accelerators()
        self.old_reduction_accelerators = _acc.get_reduction_accelerators()
        if self.backend == 'device':
            _acc.set_routine_accelerators(['cub'])
            _acc.set_reduction_accelerators([])
        elif self.backend == 'block':
            _acc.set_routine_accelerators([])
            _acc.set_reduction_accelerators(['cub'])

    def tearDown(self):
        _acc.set_routine_accelerators(self.old_routine_accelerators)
        _acc.set_reduction_accelerators(self.old_reduction_accelerators)

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1E-5)
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
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.min(axis=axis)
        # ...then perform the actual computation
        return a.min(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_cub_min_empty_axis(self, xp, dtype, contiguous_check=False):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.min(axis=())

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1E-5)
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
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.max(axis=axis)
        # ...then perform the actual computation
        return a.max(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_cub_max_empty_axis(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.max(axis=())


# This class compares unaccelerated reduction results against NumPy's
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
    'order': ('C', 'F'),
}))
@testing.gpu
class TestUnacceleratedReduction(unittest.TestCase):

    def setUp(self):
        self.old_accelerators = _acc.get_routine_accelerators()
        _acc.set_routine_accelerators([])
        # also avoid fallback to CUB via the general reduction kernel
        self.old_reduction_accelerators = _acc.get_reduction_accelerators()
        _acc.set_reduction_accelerators([])

    def tearDown(self):
        _acc.set_routine_accelerators(self.old_accelerators)
        _acc.set_reduction_accelerators(self.old_reduction_accelerators)

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_unaccelerated_min(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.min(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_unaccelerated_min_empty_axis(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.min(axis=())

    @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_unaccelerated_max(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.max(axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_unaccelerated_max_empty_axis(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype, order=self.order)
        return a.max(axis=())
