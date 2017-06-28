import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestArrayCopyAndView(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_view(self, xp):
        a = testing.shaped_arange((4,), xp, dtype=numpy.float32)
        b = a.view(dtype=numpy.int32)
        b[:] = 0
        return a

    @testing.numpy_cupy_array_equal()
    def test_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.flatten()

    @testing.numpy_cupy_array_equal()
    def test_flatten_copied(self, xp):
        a = testing.shaped_arange((4,), xp)
        b = a.flatten()
        a[:] = 1
        return b

    @testing.numpy_cupy_array_equal()
    def test_transposed_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.flatten()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(1)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_with_numpy_scalar_ndarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(numpy.ones((), dtype=dtype))
        return a

    @testing.for_all_dtypes()
    def test_fill_with_numpy_nonscalar_ndarray(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        with self.assertRaises(ValueError):
            a.fill(numpy.ones((1,), dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = a.transpose(2, 0, 1)
        b.fill(1)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='src_dtype', no_complex=True)
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype(self, xp, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return a.astype(dst_dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_all_dtypes(name='dst_dtype')
    def test_astype_type(self, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, src_dtype)
        b = a.astype(dst_dtype, order=order)
        a_cpu = testing.shaped_arange((2, 3, 4), numpy, src_dtype)
        b_cpu = a_cpu.astype(dst_dtype, order=order)
        self.assertEqual(b.dtype.type, b_cpu.dtype.type)

    @testing.for_orders('CAK')
    @testing.for_all_dtypes()
    def test_astype_type_c_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = a.astype(dtype, order=order, copy=False)
        self.assertTrue(b is a)

    @testing.for_orders('FAK')
    @testing.for_all_dtypes()
    def test_astype_type_f_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        a = cupy.asfortranarray(a)
        b = a.astype(dtype, order=order, copy=False)
        self.assertTrue(b is a)

    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype_strides(self, xp, src_dtype, dst_dtype):
        src = xp.empty((1, 2, 3), dtype=src_dtype)
        return numpy.array(src.astype(dst_dtype, order='K').strides)

    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype_strides_negative(self, xp, src_dtype, dst_dtype):
        src = xp.empty((2, 3), dtype=src_dtype)[::-1, :]
        return numpy.array(src.astype(dst_dtype, order='K').strides)

    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype_strides_swapped(self, xp, src_dtype, dst_dtype):
        src = xp.swapaxes(xp.empty((2, 3, 4), dtype=src_dtype), 1, 0)
        return numpy.array(src.astype(dst_dtype, order='K').strides)

    @testing.for_all_dtypes(name='src_dtype')
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype_strides_broadcast(self, xp, src_dtype, dst_dtype):
        src, _ = xp.broadcast_arrays(xp.empty((2,), dtype=src_dtype),
                                     xp.empty((2, 3, 2), dtype=src_dtype))
        return numpy.array(src.astype(dst_dtype, order='K').strides)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal2(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)
