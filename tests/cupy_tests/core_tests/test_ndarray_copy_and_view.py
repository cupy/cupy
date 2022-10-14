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


class TestView:

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
    @testing.with_requires('numpy>=1.23')
    def test_view_f_contiguous(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 2, 2), xp, dtype=numpy.float32)
            a = a.T
            with pytest.raises(ValueError):
                a.view(dtype=dtype)

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
        ('C', (0,)),
        ('C', (1, 3)),
        ('C', (3, 1)),
    ], ids=str)
    @testing.numpy_cupy_equal()
    def test_view_flags_smaller(self, xp, order, shape):
        a = xp.zeros(shape, numpy.int32, order)
        b = a.view(numpy.int16)
        return b.flags.c_contiguous, b.flags.f_contiguous, b.flags.owndata

    @pytest.mark.parametrize(('order', 'shape'), [
        ('F', (3, 5)),
    ], ids=str)
    @testing.with_requires('numpy>=1.23')
    def test_view_flags_smaller_invalid(self, order, shape):
        for xp in (numpy, cupy):
            a = xp.zeros(shape, numpy.int32, order)
            with pytest.raises(ValueError):
                a.view(numpy.int16)

    @pytest.mark.parametrize(('order', 'shape'), [
        ('C', (6,)),
        ('C', (3, 10)),
        ('C', (0,)),
        ('C', (1, 6)),
        ('C', (3, 2)),
    ], ids=str)
    @testing.numpy_cupy_equal()
    def test_view_flags_larger(self, xp, order, shape):
        a = xp.zeros(shape, numpy.int16, order)
        b = a.view(numpy.int32)
        return b.flags.c_contiguous, b.flags.f_contiguous, b.flags.owndata

    @pytest.mark.parametrize(('order', 'shape'), [
        ('F', (6, 5)),
        ('F', (2, 3)),
    ], ids=str)
    @testing.with_requires('numpy>=1.23')
    def test_view_flags_larger_invalid(self, order, shape):
        for xp in (numpy, cupy):
            a = xp.zeros(shape, numpy.int16, order)
            with pytest.raises(ValueError):
                a.view(numpy.int32)

    @testing.with_requires('numpy>=1.23')
    @testing.numpy_cupy_array_equal()
    def test_view_smaller_dtype_multiple(self, xp):
        # x is non-contiguous
        x = xp.arange(10, dtype=xp.int32)[::2]
        with pytest.raises(ValueError):
            x.view(xp.int16)
        return x[:, xp.newaxis].view(xp.int16)

    @testing.with_requires('numpy>=1.23')
    @testing.numpy_cupy_array_equal()
    def test_view_smaller_dtype_multiple2(self, xp):
        # x is non-contiguous, and stride[-1] != 0
        x = xp.ones((3, 4), xp.int32)[:, :1:2]
        return x.view(xp.int16)

    @testing.with_requires('numpy>=1.23')
    @testing.numpy_cupy_array_equal()
    def test_view_larger_dtype_multiple(self, xp):
        # x is non-contiguous in the first dimension, contiguous in the last
        x = xp.arange(20, dtype=xp.int16).reshape(10, 2)[::2, :]
        return x.view(xp.int32)

    @testing.with_requires('numpy>=1.23')
    @testing.numpy_cupy_array_equal()
    def test_view_non_c_contiguous(self, xp):
        # x is contiguous in axis=-1, but not C-contiguous in other axes
        x = xp.arange(2 * 3 * 4, dtype=xp.int8).reshape(
            2, 3, 4).transpose(1, 0, 2)
        return x.view(xp.int16)

    @testing.numpy_cupy_array_equal()
    def test_view_larger_dtype_zero_sized(self, xp):
        x = xp.ones((3, 20), xp.int16)[:0, ::2]
        return x.view(xp.int32)


class TestArrayCopy:

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


class TestArrayFlatten:

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
    def test_flatten_transposed(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.flatten()

    @testing.for_orders('CFAK')
    @testing.numpy_cupy_array_equal()
    def test_flatten_order(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.flatten(order)

    @testing.for_orders('CFAK')
    @testing.numpy_cupy_array_equal()
    def test_flatten_order_copied(self, xp, order):
        a = testing.shaped_arange((4,), xp)
        b = a.flatten(order=order)
        a[:] = 1
        return b

    @testing.for_orders('CFAK')
    @testing.numpy_cupy_array_equal()
    def test_flatten_order_transposed(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.flatten(order=order)


class TestArrayFill:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(1)
        return a

    @testing.for_all_dtypes_combination(('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_fill_with_numpy_scalar_ndarray(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        a.fill(numpy.ones((), dtype=dtype2))
        return a

    @testing.for_all_dtypes_combination(('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_fill_with_cupy_scalar_ndarray(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        b = xp.ones((), dtype=dtype2)

        # `numpy.can_cast` returns `True` for `from` which is a scalar or array
        # scalar that can be safely cast even if cast does not follow the
        # given casting rule. However, the similar behavior is not trivial for
        # CuPy arrays as it requires synchronization.
        b_np = cupy.asnumpy(b)
        if (
            numpy.can_cast(b_np, a.dtype)
            and not numpy.can_cast(b_np.dtype, a.dtype)
        ):
            return xp.array([])  # Skip a combination

        a.fill(b)
        return a

    @testing.for_all_dtypes()
    def test_fill_with_nonscalar_ndarray(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                a.fill(xp.ones((1,), dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = a.transpose(2, 0, 1)
        b.fill(1)
        return b


class TestArrayAsType:

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


class TestArrayDiagonal:

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


class C_cp(cupy.ndarray):

    def __new__(cls, *args, info=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


class C_np(numpy.ndarray):

    def __new__(cls, *args, info=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


class TestSubclassArrayView:

    def test_view_casting(self):
        for xp, C in [(numpy, C_np), (cupy, C_cp)]:
            a = xp.arange(5, dtype='i').view('f')
            assert type(a) is xp.ndarray
            assert a.dtype == xp.float32

            a = xp.arange(5, dtype='i').view(dtype='f')
            assert type(a) is xp.ndarray
            assert a.dtype == xp.float32

            with pytest.raises(TypeError):
                xp.arange(5, dtype='i').view('f', dtype='f')

            a = xp.arange(5, dtype='i').view(C)
            assert type(a) is C
            assert a.dtype == xp.int32
            assert a.info is None

            a = xp.arange(5, dtype='i').view(type=C)
            assert type(a) is C
            assert a.dtype == xp.int32
            assert a.info is None

            # When an instance of ndarray's subclass is supplied to `dtype`,
            # view() interprets it as if it is supplied to `type`
            a = xp.arange(5, dtype='i').view(dtype=C)
            assert type(a) is C
            assert a.dtype == xp.int32
            assert a.info is None

            with pytest.raises(TypeError):
                xp.arange(5).view('f', C, type=C)

        with pytest.raises(ValueError):
            cupy.arange(5).view(type=numpy.ndarray)
