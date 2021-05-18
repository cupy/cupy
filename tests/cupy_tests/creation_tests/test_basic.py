import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp, dtype, order):
        a = xp.empty((2, 3, 4), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_huge_size(self):
        a = cupy.empty((1024, 2048, 1024), dtype='b')
        a.fill(123)
        assert (a == 123).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_huge_size_fill0(self):
        a = cupy.empty((1024, 2048, 1024), dtype='b')
        a.fill(0)
        assert (a == 0).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_scalar(self, xp, dtype, order):
        a = xp.empty((), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.with_requires('numpy>=1.20')
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_scalar_none(self, xp, dtype, order):
        with testing.assert_warns(DeprecationWarning):
            a = xp.empty(None, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_int(self, xp, dtype, order):
        a = xp.empty(3, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_int_huge_size(self):
        a = cupy.empty(2 ** 31, dtype='b')
        a.fill(123)
        assert (a == 123).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_int_huge_size_fill0(self):
        a = cupy.empty(2 ** 31, dtype='b')
        a.fill(0)
        assert (a == 0).all()
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order)
        b.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_contiguity(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order)
        b.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_contiguity2(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order=order)
        b.fill(0)
        if order in ['c', 'C']:
            assert b.flags.c_contiguous
        else:
            assert b.flags.f_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_contiguity3(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = xp.empty_like(a, order=order)
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
    def test_empty_like_K_strides(self, dtype):
        # test strides that are both non-contiguous and non-descending
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = a[:, ::2, :].swapaxes(0, 1)
        b = numpy.empty_like(a, order='K')
        b.fill(0)

        # GPU case
        ag = testing.shaped_arange((2, 3, 4), cupy, dtype)
        ag = ag[:, ::2, :].swapaxes(0, 1)
        bg = cupy.empty_like(ag, order='K')
        bg.fill(0)

        # make sure NumPy and CuPy strides agree
        assert b.strides == bg.strides
        return

    @testing.with_requires('numpy>=1.19')
    @testing.for_all_dtypes()
    def test_empty_like_invalid_order(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.empty_like(a, order='Q')

    def test_empty_like_subok(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        with pytest.raises(TypeError):
            cupy.empty_like(a, subok=True)

    @testing.for_CF_orders()
    def test_empty_zero_sized_array_strides(self, order):
        a = numpy.empty((1, 0, 2), dtype='d', order=order)
        b = cupy.empty((1, 0, 2), dtype='d', order=order)
        assert b.strides == a.strides

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_eye(self, xp, dtype, order):
        return xp.eye(5, 4, 1, dtype, order=order)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_identity(self, xp, dtype):
        return xp.identity(4, dtype)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros(self, xp, dtype, order):
        return xp.zeros((2, 3, 4), dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_scalar(self, xp, dtype, order):
        return xp.zeros((), dtype=dtype, order=order)

    @testing.with_requires('numpy>=1.20')
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_scalar_none(self, xp, dtype, order):
        with testing.assert_warns(DeprecationWarning):
            return xp.zeros(None, dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_int(self, xp, dtype, order):
        return xp.zeros(3, dtype=dtype, order=order)

    @testing.for_CF_orders()
    def test_zeros_strides(self, order):
        a = numpy.zeros((2, 3), dtype='d', order=order)
        b = cupy.zeros((2, 3), dtype='d', order=order)
        assert b.strides == a.strides

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_like(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.zeros_like(a, order=order)

    def test_zeros_like_subok(self):
        a = cupy.ndarray((2, 3, 4))
        with pytest.raises(TypeError):
            cupy.zeros_like(a, subok=True)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones(self, xp, dtype, order):
        return xp.ones((2, 3, 4), dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones_like(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a, order=order)

    def test_ones_like_subok(self):
        a = cupy.ndarray((2, 3, 4))
        with pytest.raises(TypeError):
            cupy.ones_like(a, subok=True)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full(self, xp, dtype, order):
        return xp.full((2, 3, 4), 1, dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_default_dtype(self, xp, dtype, order):
        return xp.full((2, 3, 4), xp.array(1, dtype=dtype), order=order)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_default_dtype_cpu_input(self, xp, dtype):
        return xp.full((2, 3, 4), numpy.array(1, dtype=dtype))

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_like(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.full_like(a, 1, order=order)

    def test_full_like_subok(self):
        a = cupy.ndarray((2, 3, 4))
        with pytest.raises(TypeError):
            cupy.full_like(a, 1, subok=True)


@testing.parameterize(
    *testing.product({
        'shape': [4, (4, ), (4, 2), (4, 2, 3), (5, 4, 2, 3)],
    })
)
@testing.gpu
class TestBasicReshape(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_reshape(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_empty_like_reshape_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.empty_like(a, shape=self.shape)
        b.fill(0)
        c = cupy.empty(self.shape, order=order, dtype=dtype)
        c.fill(0)

        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_reshape_contiguity(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_empty_like_reshape_contiguity_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        c = cupy.empty(self.shape)
        c.fill(0)
        if order in ['f', 'F']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_reshape_contiguity2(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order=order, shape=self.shape)
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
    def test_empty_like_reshape_contiguity2_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        a = cupy.asfortranarray(a)
        b = cupy.empty_like(a, order=order, shape=self.shape)
        b.fill(0)
        c = cupy.empty(self.shape)
        c.fill(0)
        shape = self.shape if not numpy.isscalar(self.shape) else (self.shape,)
        if (order in ['c', 'C'] or
                (order in ['k', 'K', None] and len(shape) != a.ndim)):
            assert b.flags.c_contiguous
        else:
            assert b.flags.f_contiguous
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_reshape_contiguity3(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = xp.empty_like(a, order=order, shape=self.shape)
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
    def test_empty_like_reshape_contiguity3_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = cupy.empty_like(a, order=order, shape=self.shape)
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

        c = cupy.zeros(self.shape)
        c.fill(0)
        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_all_dtypes()
    def test_empty_like_K_strides_reshape(self, dtype):
        # test strides that are both non-contiguous and non-descending
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = a[:, ::2, :].swapaxes(0, 1)
        b = numpy.empty_like(a, order='K', shape=self.shape)
        b.fill(0)

        # GPU case
        ag = testing.shaped_arange((2, 3, 4), cupy, dtype)
        ag = ag[:, ::2, :].swapaxes(0, 1)
        bg = cupy.empty_like(ag, order='K', shape=self.shape)
        bg.fill(0)

        # make sure NumPy and CuPy strides agree
        assert b.strides == bg.strides
        return

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_like_reshape(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.zeros_like(a, order=order, shape=self.shape)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_zeros_like_reshape_cupy_only(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.zeros_like(a, shape=self.shape)
        c = cupy.zeros(self.shape, order=order, dtype=dtype)

        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones_like_reshape(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a, order=order, shape=self.shape)

    @testing.for_all_dtypes()
    def test_ones_like_reshape_cupy_only(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.ones_like(a, shape=self.shape)
        c = cupy.ones(self.shape, dtype=dtype)

        testing.assert_array_equal(b, c)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_like_reshape(self, xp, dtype, order):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.full_like(a, 1, order=order, shape=self.shape)

    @testing.for_all_dtypes()
    def test_full_like_reshape_cupy_only(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.full_like(a, 1, shape=self.shape)
        c = cupy.full(self.shape, 1, dtype=dtype)

        testing.assert_array_equal(b, c)
