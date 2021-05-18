import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing._helper import _wraps_partial
import cupyx


def numpy_cupyx_array_equal(target_func, name='func'):
    _mod = (cupy, numpy)

    _numpy_funcs = {
        'empty': numpy.empty,
        'empty_like': numpy.empty_like,
        'zeros': numpy.zeros,
        'zeros_like': numpy.zeros_like,
    }

    _cupy_funcs = {
        'empty': cupyx.empty_pinned,
        'empty_like': cupyx.empty_like_pinned,
        'zeros': cupyx.zeros_pinned,
        'zeros_like': cupyx.zeros_like_pinned,
    }

    def _get_test_func(xp, func):
        if xp is numpy:
            return _numpy_funcs[func]
        elif xp is cupy:
            return _cupy_funcs[func]
        else:
            assert False

    def _check_pinned_mem_used(a, xp):
        if xp is cupy:
            assert isinstance(a.base, cupy.cuda.PinnedMemoryPointer)
            assert a.base.ptr == a.ctypes.data

    def decorator(impl):
        @_wraps_partial(impl, name)
        def test_func(self, *args, **kw):
            out = []
            for xp in _mod:
                func = _get_test_func(xp, target_func)
                kw[name] = func
                a = impl(self, *args, **kw)
                _check_pinned_mem_used(a, xp)
                out.append(a)
            numpy.testing.assert_array_equal(*out)
        return test_func
    return decorator


# test_empty_scalar_none is removed
# test_zeros_scalar_none is removed
class TestBasic(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty')
    def test_empty(self, dtype, order, func):
        a = func((2, 3, 4), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_huge_size(self):
        a = cupyx.empty_pinned((1024, 2048, 1024), dtype='b')
        a.fill(123)
        assert (a == 123).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_huge_size_fill0(self):
        a = cupyx.empty_pinned((1024, 2048, 1024), dtype='b')
        a.fill(0)
        assert (a == 0).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty')
    def test_empty_scalar(self, dtype, order, func):
        a = func((), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty')
    def test_empty_int(self, dtype, order, func):
        a = func(3, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_int_huge_size(self):
        a = cupyx.empty_pinned(2 ** 31, dtype='b')
        a.fill(123)
        assert (a == 123).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_int_huge_size_fill0(self):
        a = cupyx.empty_pinned(2 ** 31, dtype='b')
        a.fill(0)
        assert (a == 0).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = func(a, order=order)
        b.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_contiguity(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = func(a, order=order)
        b.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_contiguity2(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = numpy.asfortranarray(a)
        b = func(a, order=order)
        b.fill(0)
        if order in ['c', 'C']:
            assert b.flags.c_contiguous
        else:
            assert b.flags.f_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_contiguity3(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = func(a, order=order)
        b.fill(0)
        if order in ['k', 'K', None]:
            assert not b.flags.c_contiguous
            assert not b.flags.f_contiguous
        elif order in ['f', 'F']:
            assert not b.flags.c_contiguous
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
            assert not b.flags.f_contiguous
        return b

    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_K_strides(self, dtype):
        # test strides that are both non-contiguous and non-descending;
        # also test accepting cupy.ndarray
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = a[:, ::2, :].swapaxes(0, 1)
        b = numpy.empty_like(a, order='K')
        b.fill(0)

        # GPU case
        ag = testing.shaped_arange((2, 3, 4), cupy, dtype)
        ag = ag[:, ::2, :].swapaxes(0, 1)
        bg = cupyx.empty_like_pinned(ag, order='K')
        bg.fill(0)

        # make sure NumPy and CuPy strides agree
        assert b.strides == bg.strides

    @testing.with_requires('numpy>=1.19')
    @testing.for_all_dtypes()
    def test_empty_like_invalid_order(self, dtype):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        with pytest.raises(ValueError):
            cupyx.empty_like_pinned(a, order='Q')

    def test_empty_like_subok(self):
        a = testing.shaped_arange((2, 3, 4), numpy)
        with pytest.raises(TypeError):
            cupyx.empty_like_pinned(a, subok=True)

    @testing.for_CF_orders()
    def test_empty_zero_sized_array_strides(self, order):
        a = numpy.empty((1, 0, 2), dtype='d', order=order)
        b = cupyx.empty_pinned((1, 0, 2), dtype='d', order=order)
        assert b.strides == a.strides

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='zeros')
    def test_zeros(self, dtype, order, func):
        return func((2, 3, 4), dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='zeros')
    def test_zeros_scalar(self, dtype, order, func):
        return func((), dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='zeros')
    def test_zeros_int(self, dtype, order, func):
        return func(3, dtype=dtype, order=order)

    @testing.for_CF_orders()
    def test_zeros_strides(self, order):
        a = numpy.zeros((2, 3), dtype='d', order=order)
        b = cupyx.zeros_pinned((2, 3), dtype='d', order=order)
        assert b.strides == a.strides

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='zeros_like')
    def test_zeros_like(self, dtype, order, func):
        a = numpy.ndarray((2, 3, 4), dtype=dtype)
        return func(a, order=order)

    def test_zeros_like_subok(self):
        a = numpy.ndarray((2, 3, 4))
        with pytest.raises(TypeError):
            cupyx.zeros_like_pinned(a, subok=True)


@testing.parameterize(
    *testing.product({
        'shape': [4, (4, ), (4, 2), (4, 2, 3), (5, 4, 2, 3)],
    })
)
class TestBasicReshape(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_reshape(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = func(a, order=order, shape=self.shape)
        b.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_reshape_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupyx.empty_like_pinned(a, shape=self.shape)
        b.fill(0)
        c = cupyx.empty_pinned(self.shape, order=order, dtype=dtype)
        c.fill(0)
        numpy.testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_reshape_contiguity(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = func(a, order=order, shape=self.shape)
        b.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_reshape_contiguity_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupyx.empty_like_pinned(a, order=order, shape=self.shape)
        b.fill(0)
        c = cupyx.empty_pinned(self.shape)
        c.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        numpy.testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_reshape_contiguity2(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = numpy.asfortranarray(a)
        b = func(a, order=order, shape=self.shape)
        b.fill(0)
        shape = self.shape if not numpy.isscalar(self.shape) else (self.shape,)
        if (order in ['c', 'C'] or
                (order in ['k', 'K', None] and len(shape) != a.ndim)):
            assert b.flags.c_contiguous
        else:
            assert b.flags.f_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_reshape_contiguity2_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        a = cupy.asfortranarray(a)
        b = cupyx.empty_like_pinned(a, order=order, shape=self.shape)
        b.fill(0)
        c = cupyx.empty_pinned(self.shape)
        c.fill(0)
        shape = self.shape if not numpy.isscalar(self.shape) else (self.shape,)
        if (order in ['c', 'C'] or
                (order in ['k', 'K', None] and len(shape) != a.ndim)):
            assert b.flags.c_contiguous
        else:
            assert b.flags.f_contiguous
        numpy.testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='empty_like')
    def test_empty_like_reshape_contiguity3(self, dtype, order, func):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = func(a, order=order, shape=self.shape)
        b.fill(0)
        shape = self.shape if not numpy.isscalar(self.shape) else (self.shape,)
        if len(shape) == 1:
            assert b.flags.c_contiguous
            assert b.flags.f_contiguous
        elif order in ['k', 'K', None] and len(shape) == a.ndim:
            assert not b.flags.c_contiguous
            assert not b.flags.f_contiguous
        elif order in ['f', 'F']:
            assert not b.flags.c_contiguous
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
            assert not b.flags.f_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_reshape_contiguity3_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = cupyx.empty_like_pinned(a, order=order, shape=self.shape)
        b.fill(0)
        shape = self.shape if not numpy.isscalar(self.shape) else (self.shape,)
        if len(shape) == 1:
            assert b.flags.c_contiguous
            assert b.flags.f_contiguous
        elif order in ['k', 'K', None] and len(shape) == a.ndim:
            assert not b.flags.c_contiguous
            assert not b.flags.f_contiguous
        elif order in ['f', 'F']:
            assert not b.flags.c_contiguous
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
            assert not b.flags.f_contiguous

        c = cupyx.zeros_pinned(self.shape)
        c.fill(0)
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_all_dtypes()
    @testing.gpu
    def test_empty_like_K_strides_reshape(self, dtype):
        # test strides that are both non-contiguous and non-descending
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = a[:, ::2, :].swapaxes(0, 1)
        b = cupyx.empty_like_pinned(a, order='K', shape=self.shape)
        b.fill(0)

        # GPU case
        ag = testing.shaped_arange((2, 3, 4), cupy, dtype)
        ag = ag[:, ::2, :].swapaxes(0, 1)
        bg = cupyx.empty_like_pinned(ag, order='K', shape=self.shape)
        bg.fill(0)

        # make sure NumPy and CuPy strides agree
        assert b.strides == bg.strides
        return

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @numpy_cupyx_array_equal(target_func='zeros_like')
    def test_zeros_like_reshape(self, dtype, order, func):
        a = numpy.ndarray((2, 3, 4), dtype=dtype)
        return func(a, order=order, shape=self.shape)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.gpu
    def test_zeros_like_reshape_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupyx.zeros_like_pinned(a, shape=self.shape)
        c = cupyx.zeros_pinned(self.shape, order=order, dtype=dtype)
        numpy.testing.assert_array_equal(b, c)
