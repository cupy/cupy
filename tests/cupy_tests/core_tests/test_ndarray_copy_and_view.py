import contextlib

import numpy
import pytest

import cupy
from cupy import testing
from cupy import _util


def astype_without_warning(x, dtype, *args, **kwargs):
    dtype = numpy.dtype(dtype)
    if x.dtype.kind == 'c' and dtype.kind not in ['b', 'c']:
        with testing.assert_warns(numpy.ComplexWarning):
            return x.astype(dtype, *args, **kwargs)
    else:
        return x.astype(dtype, *args, **kwargs)


@testing.gpu
class TestArrayCopyAndView:

    @testing.numpy_cupy_array_equal()
    def test_view(self, xp):
        a = testing.shaped_arange((4,), xp, dtype=numpy.float32)
        b = a.view(dtype=numpy.int32)
        b[:] = 0
        return a

    @testing.for_dtypes([numpy.int16, numpy.int64])
    @testing.numpy_cupy_array_equal()
    def test_view_itemsize(self, xp, dtype):
        a = testing.shaped_arange((4,), xp, dtype=numpy.int32)
        b = a.view(dtype=dtype)
        return b

    @testing.numpy_cupy_array_equal()
    def test_view_0d(self, xp):
        a = xp.array(1.5, dtype=numpy.float32)
        return a.view(dtype=numpy.int32)

    @testing.for_dtypes([numpy.int16, numpy.int64])
    def test_view_0d_raise(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array(3, dtype=numpy.int32)
            with pytest.raises(ValueError):
                a.view(dtype=dtype)

    @testing.for_dtypes([numpy.int16, numpy.int64])
    def test_view_non_contiguous_raise(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 2, 2), xp, dtype=numpy.int32)
            a = a.transpose(0, 2, 1)
            with pytest.raises(ValueError):
                a.view(dtype=dtype)

    @testing.for_dtypes([numpy.int16, numpy.int64])
    @testing.numpy_cupy_array_equal()
    def test_view_f_contiguous(self, dtype, xp):
        a = testing.shaped_arange((2, 2, 2), xp, dtype=numpy.float32)
        a = a.T
        with testing.assert_warns(DeprecationWarning):
            return a.view(dtype=dtype)

    def test_view_assert_divisible(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3,), xp, dtype=numpy.int32)
            with pytest.raises(ValueError):
                a.view(dtype=numpy.int64)

    @testing.for_dtypes([numpy.float32, numpy.float64])
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_view_relaxed_contiguous(self, xp, dtype):
        a = testing.shaped_arange((1, 3, 5), xp, dtype=dtype)
        a = xp.moveaxis(a, 0, 2)  # (3, 5, 1)
        b = a.view(dtype=numpy.int32)
        return b

    @pytest.mark.parametrize(('order', 'shape'), [
        ('C', (3,)),
        ('C', (3, 5)),
        ('F', (3, 5)),
        ('C', (0,)),
        ('C', (1, 3)),
        ('C', (3, 1)),
    ], ids=str)
    @testing.numpy_cupy_equal()
    def test_view_flags_smaller(self, xp, order, shape):
        a = xp.zeros(shape, numpy.int32, order)
        with contextlib.ExitStack() as stack:
            if order == 'F':
                stack.enter_context(testing.assert_warns(DeprecationWarning))
            b = a.view(numpy.int16)
        return b.flags.c_contiguous, b.flags.f_contiguous, b.flags.owndata

    @pytest.mark.parametrize(('order', 'shape'), [
        ('C', (6,)),
        ('C', (3, 10)),
        ('F', (6, 5)),
        ('C', (0,)),
        ('C', (1, 6)),
        ('F', (2, 3)),
        ('C', (3, 2)),
    ], ids=str)
    @testing.numpy_cupy_equal()
    def test_view_flags_larger(self, xp, order, shape):
        a = xp.zeros(shape, numpy.int16, order)
        with contextlib.ExitStack() as stack:
            if order == 'F':
                stack.enter_context(testing.assert_warns(DeprecationWarning))
            b = a.view(numpy.int32)
        return b.flags.c_contiguous, b.flags.f_contiguous, b.flags.owndata

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
        with pytest.raises(ValueError):
            a.fill(numpy.ones((1,), dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = a.transpose(2, 0, 1)
        b.fill(1)
        return b

    @testing.for_orders(['C', 'F', 'A', 'K', None])
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_array_equal()
    def test_astype(self, xp, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return astype_without_warning(a, dst_dtype, order=order)

    @testing.for_orders(['C', 'F', 'A', 'K', None])
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_array_equal()
    def test_astype_empty(self, xp, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 0, 4), xp, src_dtype)
        return astype_without_warning(a, dst_dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    def test_astype_type(self, src_dtype, dst_dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, src_dtype)
        b = astype_without_warning(a, dst_dtype, order=order)
        a_cpu = testing.shaped_arange((2, 3, 4), numpy, src_dtype)
        b_cpu = astype_without_warning(a_cpu, dst_dtype, order=order)
        assert b.dtype.type == b_cpu.dtype.type

    @testing.for_orders('CAK')
    @testing.for_all_dtypes()
    def test_astype_type_c_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = a.astype(dtype, order=order, copy=False)
        assert b is a

    @testing.for_orders('FAK')
    @testing.for_all_dtypes()
    def test_astype_type_f_contiguous_no_copy(self, dtype, order):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        a = cupy.asfortranarray(a)
        b = a.astype(dtype, order=order, copy=False)
        assert b is a

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_equal()
    def test_astype_strides(self, xp, src_dtype, dst_dtype):
        src = xp.empty((1, 2, 3), dtype=src_dtype)
        return astype_without_warning(src, dst_dtype, order='K').strides

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_equal()
    def test_astype_strides_negative(self, xp, src_dtype, dst_dtype):
        src = xp.empty((2, 3), dtype=src_dtype)[::-1, :]
        return astype_without_warning(src, dst_dtype, order='K').strides

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_equal()
    def test_astype_strides_swapped(self, xp, src_dtype, dst_dtype):
        src = xp.swapaxes(xp.empty((2, 3, 4), dtype=src_dtype), 1, 0)
        return astype_without_warning(src, dst_dtype, order='K').strides

    @testing.for_all_dtypes_combination(('src_dtype', 'dst_dtype'))
    @testing.numpy_cupy_equal()
    def test_astype_strides_broadcast(self, xp, src_dtype, dst_dtype):
        src, _ = xp.broadcast_arrays(xp.empty((2,), dtype=src_dtype),
                                     xp.empty((2, 3, 2), dtype=src_dtype))
        return astype_without_warning(src, dst_dtype, order='K').strides

    @testing.numpy_cupy_array_equal()
    def test_astype_boolean_view(self, xp):
        # See #4354
        a = xp.array([0, 1, 2], dtype=numpy.int8).view(dtype=numpy.bool_)
        return a.astype(numpy.int8)

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

    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int16, numpy.int64,
                         numpy.float16, numpy.float64])
    @testing.numpy_cupy_array_equal()
    def test_isinstance_numpy_copy(self, xp, dtype, order):
        a = numpy.arange(100, dtype=dtype).reshape(10, 10, order=order)
        b = xp.empty(a.shape, dtype=dtype, order=order)
        b[:] = a
        return b

    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    def test_isinstance_numpy_copy_wrong_dtype(self):
        a = numpy.arange(100, dtype=numpy.float64).reshape(10, 10)
        b = cupy.empty(a.shape, dtype=numpy.int32)
        with pytest.raises(ValueError):
            b[:] = a

    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    def test_isinstance_numpy_copy_wrong_shape(self):
        for xp in (numpy, cupy):
            a = numpy.arange(100, dtype=numpy.float64).reshape(10, 10)
            b = cupy.empty(100, dtype=a.dtype)
            with pytest.raises(ValueError):
                b[:] = a

    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    @testing.numpy_cupy_array_equal()
    def test_isinstance_numpy_copy_not_slice(self, xp):
        a = xp.arange(5, dtype=numpy.float64)
        a[a < 3] = 0
        return a

    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    def test_copy_host_to_device_view(self):
        dev = cupy.empty((10, 10), dtype=numpy.float32)[2:5, 1:8]
        host = numpy.arange(3 * 7, dtype=numpy.float32).reshape(3, 7)
        with pytest.raises(ValueError):
            dev[:] = host


@testing.parameterize(
    {'src_order': 'C'},
    {'src_order': 'F'},
)
@testing.gpu
class TestNumPyArrayCopyView:
    @pytest.mark.skipif(
        not _util.ENABLE_SLICE_COPY, reason='Special copy disabled')
    @testing.for_orders('CF')
    @testing.for_dtypes([numpy.int16, numpy.int64,
                         numpy.float16, numpy.float64])
    @testing.numpy_cupy_array_equal()
    def test_isinstance_numpy_view_copy_f(self, xp, dtype, order):
        a = numpy.arange(100, dtype=dtype).reshape(
            10, 10, order=self.src_order)
        a = a[2:5, 1:8]
        b = xp.empty(a.shape, dtype=dtype, order=order)
        b[:] = a
        return b
