import unittest

import numpy
import pytest

import cupy
import cupy._core._accelerator as _acc
from cupy._core import _cub_reduction
from cupy import testing


@testing.gpu
class TestSearch(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_external_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_argmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_external_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_tie(self, xp, dtype):
        a = xp.array([0, 5, 2, 3, 4, 5], dtype)
        return a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    def test_argmax_zero_size(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                a.argmax()

    @testing.for_all_dtypes(no_complex=True)
    def test_argmax_zero_size_axis0(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                a.argmax(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmax_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_argmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_external_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.argmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_external_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.argmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_tie(self, xp, dtype):
        a = xp.array([0, 1, 2, 3, 0, 5], dtype)
        return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    def test_argmin_zero_size(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                return a.argmin()

    @testing.for_all_dtypes(no_complex=True)
    def test_argmin_zero_size_axis0(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                a.argmin(axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_argmin_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return a.argmin(axis=1)


# TODO(leofang): remove this once CUDA 9.0 is dropped
def _skip_cuda90(dtype):
    ver = cupy.cuda.runtime.runtimeGetVersion()
    if dtype == cupy.float16 and ver == 9000:
        pytest.skip('CUB does not support fp16 on CUDA 9.0')


# This class compares CUB results against NumPy's
# TODO(leofang): test axis after support is added
@testing.parameterize(*testing.product({
    'shape': [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
    'order_and_axis': (('C', -1), ('C', None), ('F', 0), ('F', None)),
    'backend': ('device', 'block'),
}))
@testing.gpu
@unittest.skipUnless(cupy.cuda.cub.available, 'The CUB routine is not enabled')
class TestCubReduction(unittest.TestCase):

    def setUp(self):
        self.order, self.axis = self.order_and_axis
        self.old_routine_accelerators = _acc.get_routine_accelerators()
        self.old_reduction_accelerators = _acc.get_reduction_accelerators()
        if self.backend == 'device':
            if self.axis is not None:
                raise unittest.SkipTest('does not support')
            _acc.set_routine_accelerators(['cub'])
            _acc.set_reduction_accelerators([])
        elif self.backend == 'block':
            _acc.set_routine_accelerators([])
            _acc.set_reduction_accelerators(['cub'])

    def tearDown(self):
        _acc.set_routine_accelerators(self.old_routine_accelerators)
        _acc.set_reduction_accelerators(self.old_reduction_accelerators)

    @testing.for_dtypes('bhilBHILefdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_cub_argmin(self, xp, dtype):
        _skip_cuda90(dtype)
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.order == 'C':
            a = xp.ascontiguousarray(a)
        else:
            a = xp.asfortranarray(a)

        if xp is numpy:
            return a.argmin(axis=self.axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if self.backend == 'device':
            func_name = 'cupy._core._routines_statistics.cub.'
            func_name += 'device_reduce'
            with testing.AssertFunctionIsCalled(func_name, return_value=ret):
                a.argmin(axis=self.axis)
        elif self.backend == 'block':
            # this is the only function we can mock; the rest is cdef'd
            func_name = 'cupy._core._cub_reduction.'
            func_name += '_SimpleCubReductionKernel_get_cached_function'
            func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
            if self.axis is not None and len(self.shape) > 1:
                times_called = 1  # one pass
            else:
                times_called = 2  # two passes
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.argmin(axis=self.axis)
        # ...then perform the actual computation
        return a.argmin(axis=self.axis)

    @testing.for_dtypes('bhilBHILefdFD')
    @testing.numpy_cupy_allclose(rtol=1E-5, contiguous_check=False)
    def test_cub_argmax(self, xp, dtype):
        _skip_cuda90(dtype)
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.order == 'C':
            a = xp.ascontiguousarray(a)
        else:
            a = xp.asfortranarray(a)

        if xp is numpy:
            return a.argmax(axis=self.axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if self.backend == 'device':
            func_name = 'cupy._core._routines_statistics.cub.'
            func_name += 'device_reduce'
            with testing.AssertFunctionIsCalled(func_name, return_value=ret):
                a.argmax(axis=self.axis)
        elif self.backend == 'block':
            # this is the only function we can mock; the rest is cdef'd
            func_name = 'cupy._core._cub_reduction.'
            func_name += '_SimpleCubReductionKernel_get_cached_function'
            func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
            if self.axis is not None and len(self.shape) > 1:
                times_called = 1  # one pass
            else:
                times_called = 2  # two passes
            with testing.AssertFunctionIsCalled(
                    func_name, wraps=func, times_called=times_called):
                a.argmax(axis=self.axis)
        # ...then perform the actual computation
        return a.argmax(axis=self.axis)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['argmin', 'argmax'],
    'is_module': [True, False],
    'shape': [(3, 4), ()],
}))
class TestArgMinMaxDtype(unittest.TestCase):

    @testing.for_dtypes(
        dtypes=[numpy.int8, numpy.int16, numpy.int32, numpy.int64],
        name='result_dtype')
    @testing.for_all_dtypes(name='in_dtype')
    def test_argminmax_dtype(self, in_dtype, result_dtype):
        a = testing.shaped_random(self.shape, cupy, in_dtype)
        if self.is_module:
            func = getattr(cupy, self.func)
            y = func(a, dtype=result_dtype)
        else:
            func = getattr(a, self.func)
            y = func(dtype=result_dtype)
        assert y.shape == ()
        assert y.dtype == result_dtype


@testing.parameterize(
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (4,),      'x_shape': (2, 3, 4), 'y_shape': (2, 3, 4)},
    {'cond_shape': (2, 3, 4), 'x_shape': (2, 3, 4), 'y_shape': (3, 4)},
    {'cond_shape': (3, 4),    'x_shape': (2, 3, 4), 'y_shape': (4,)},
)
@testing.gpu
class TestWhereTwoArrays(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['cond_type', 'x_type', 'y_type'])
    @testing.numpy_cupy_allclose()
    def test_where_two_arrays(self, xp, cond_type, x_type, y_type):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        # Almost all values of a matrix `shaped_random` makes are not zero.
        # To make a sparse matrix, we need multiply `m`.
        cond = testing.shaped_random(self.cond_shape, xp, cond_type) * m
        x = testing.shaped_random(self.x_shape, xp, x_type, seed=0)
        y = testing.shaped_random(self.y_shape, xp, y_type, seed=1)
        return xp.where(cond, x, y)


@testing.parameterize(
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (4,)},
    {'cond_shape': (2, 3, 4)},
    {'cond_shape': (3, 4)},
)
@testing.gpu
class TestWhereCond(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_where_cond(self, xp, dtype):
        m = testing.shaped_random(self.cond_shape, xp, xp.bool_)
        cond = testing.shaped_random(self.cond_shape, xp, dtype) * m
        return xp.where(cond)


@testing.gpu
class TestWhereError(unittest.TestCase):

    def test_one_argument(self):
        for xp in (numpy, cupy):
            cond = testing.shaped_random((3, 4), xp, dtype=xp.bool_)
            x = testing.shaped_random((2, 3, 4), xp, xp.int32)
            with pytest.raises(ValueError):
                xp.where(cond, x)


@testing.parameterize(
    {'array': numpy.random.randint(0, 2, (20,))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
@testing.gpu
class TestNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.nonzero(array)


@testing.parameterize(
    {'array': numpy.array(0)},
    {'array': numpy.array(1)},
)
@testing.gpu
@testing.with_requires('numpy>=1.17.0')
class TestNonzeroZeroDimension(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        with testing.assert_warns(DeprecationWarning):
            return xp.nonzero(array)


@testing.parameterize(
    {'array': numpy.random.randint(0, 2, (20,))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.array(0)},
    {'array': numpy.array(1)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
@testing.gpu
class TestFlatNonzero(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_flatnonzero(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.flatnonzero(array)


@testing.parameterize(
    {'array': numpy.random.randint(0, 2, (20,))},
    {'array': numpy.random.randn(3, 2, 4)},
    {'array': numpy.empty((0,))},
    {'array': numpy.empty((0, 2))},
    {'array': numpy.empty((0, 2, 0))},
)
@testing.gpu
class TestArgwhere(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_argwhere(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        return xp.argwhere(array)


@testing.parameterize(
    {'value': 0},
    {'value': 3},
)
@testing.gpu
@testing.with_requires('numpy>=1.18')
class TestArgwhereZeroDimension(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_argwhere(self, xp, dtype):
        array = xp.array(self.value, dtype=dtype)
        return xp.argwhere(array)


@testing.gpu
class TestNanArgMin(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmin_nan2(self, xp, dtype):
        a = xp.array([float('nan'), float('nan'), -1, 1], dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmin_nan3(self, xp, dtype):
        a = xp.array([float('nan'), float('nan'), -1, 1, 1.0, -2.0], dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmin_nan4(self, xp, dtype):
        a = xp.array([-1, 1, 1.0, -2.0, float('nan'), float('nan')],
                     dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmin_nan5(self, xp, dtype):
        a = xp.array([-1, 1, 1.0, -2.0, float('nan'), float('nan'), -1, 1],
                     dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanargmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmin(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmin(a, axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_tie(self, xp, dtype):
        a = xp.array([0, 5, 2, 3, 4, 5], dtype)
        return xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    def test_nanargmin_zero_size(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                xp.nanargmin(a)

    @testing.for_all_dtypes(no_complex=True)
    def test_nanargmin_zero_size_axis0(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                return xp.nanargmin(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmin_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return xp.nanargmin(a, axis=1)


@testing.gpu
class TestNanArgMax(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), -1, 1], dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmax_nan2(self, xp, dtype):
        a = xp.array([float('nan'), float('nan'), -1, 1], dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmax_nan3(self, xp, dtype):
        a = xp.array([float('nan'), float('nan'), -1, 1, 1.0, -2.0], dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmax_nan4(self, xp, dtype):
        a = xp.array([-1, 1, 1.0, -2.0, float('nan'), float('nan')],
                     dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_nanargmax_nan5(self, xp, dtype):
        a = xp.array([-1, 1, 1.0, -2.0, float('nan'), float('nan'), -1, 1],
                     dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanargmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmax(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanargmax(a, axis=2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_tie(self, xp, dtype):
        a = xp.array([0, 5, 2, 3, 4, 5], dtype)
        return xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    def test_nanargmax_zero_size(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                xp.nanargmax(a)

    @testing.for_all_dtypes(no_complex=True)
    def test_nanargmax_zero_size_axis0(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0, 1), xp, dtype)
            with pytest.raises(ValueError):
                return xp.nanargmax(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_nanargmax_zero_size_axis1(self, xp, dtype):
        a = testing.shaped_random((0, 1), xp, dtype)
        return xp.nanargmax(a, axis=1)


@testing.gpu
@testing.parameterize(*testing.product(
    {'bins': [
        [],
        [0, 1, 2, 4, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0.0, 1.0, 2.5, 4.0, 10.0],
        [-1.0, 1.0, 2.5, 4.0, 20.0],
        [1.5, 2.5, 4.0, 6.0],
        [float('-inf'), 1.5, 2.5, 4.0, 6.0],
        [1.5, 2.5, 4.0, 6.0, float('inf')],
        [float('-inf'), 1.5, 2.5, 4.0, 6.0, float('inf')],
        [0.0, 1.0, 1.0, 4.0, 4.0, 10.0],
        [0.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 10.0],
    ],
        'side': ['left', 'right'],
        'shape': [(), (10,), (6, 3, 3)]})
)
class TestSearchSorted(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_searchsorted(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        bins = xp.array(self.bins)
        y = xp.searchsorted(bins, x, side=self.side)
        return y,


@testing.gpu
@testing.parameterize(
    {'side': 'left'},
    {'side': 'right'})
class TestSearchSortedNanInf(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_nanbins(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        bins = xp.array([0, 1, 2, 4, 10, float('nan')])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_nan(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[5] = float('nan')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_nan_last(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[-1] = float('nan')
        bins = xp.array([0, 1, 2, 4, float('nan')])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_nan_last_repeat(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[-1] = float('nan')
        bins = xp.array([0, 1, 2, float('nan'), float('nan')])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_all_nans(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[-1] = float('nan')
        bins = xp.array([float('nan'), float('nan'), float('nan'),
                         float('nan'), float('nan')])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_inf(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[5] = float('inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,

    @testing.numpy_cupy_array_equal()
    def test_searchsorted_minf(self, xp):
        x = testing.shaped_arange((10,), xp, xp.float64)
        x[5] = float('-inf')
        bins = xp.array([0, 1, 2, 4, 10])
        y = xp.searchsorted(bins, x, side=self.side)
        return y,


@testing.gpu
class TestSearchSortedInvalid(unittest.TestCase):

    # Cant test unordered bins due to numpy undefined
    # behavior for searchsorted

    def test_searchsorted_ndbins(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((10,), xp, xp.float64)
            bins = xp.array([[10, 4], [2, 1], [7, 8]])
            with pytest.raises(ValueError):
                xp.searchsorted(bins, x)


@testing.gpu
class TestSearchSortedWithSorter(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_sorter(self, xp):
        x = testing.shaped_arange((12,), xp, xp.float64)
        bins = xp.array([10, 4, 2, 1, 8])
        sorter = xp.array([3, 2, 1, 4, 0])
        y = xp.searchsorted(bins, x, sorter=sorter)
        return y,

    def test_invalid_sorter(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((12,), xp, xp.float64)
            bins = xp.array([10, 4, 2, 1, 8])
            sorter = xp.array([0])
            with pytest.raises(ValueError):
                xp.searchsorted(bins, x, sorter=sorter)

    def test_nonint_sorter(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((12,), xp, xp.float64)
            bins = xp.array([10, 4, 2, 1, 8])
            sorter = xp.array([], dtype=xp.float64)
            with pytest.raises(TypeError):
                xp.searchsorted(bins, x, sorter=sorter)
