import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing
import numpy


@testing.gpu
class TestFromData(unittest.TestCase):

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array(self, xp, dtype, order):
        return xp.array([[1, 2, 3], [2, 3, 4]], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_array_equal()
    def test_array_from_numpy_broad_cast(self, xp, dtype, order):
        a = testing.shaped_arange((2, 1, 4), numpy, dtype)
        a = numpy.broadcast_to(a, (2, 3, 4))
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy_is_copied(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype(self, xp, dtype1, dtype2, order):
        # complex to real makes no sense
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=dtype2, order=order)

    @testing.for_orders('CFAK')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype_being_none(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.array(a, dtype=None, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_no_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_f_contiguous_input(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype, order='F')
        b = xp.array(a, copy=False, order=order)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_f_contiguous_output(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order='F')
        assert b.flags.f_contiguous
        return b

    @testing.multi_gpu(2)
    def test_array_multi_device(self):
        with cuda.Device(0):
            x = testing.shaped_arange((2, 3, 4), cupy, dtype='f')
        with cuda.Device(1):
            y = cupy.array(x)
        assert isinstance(y, cupy.ndarray)
        assert x is not y  # Do copy
        assert int(x.device) == 0
        assert int(y.device) == 1
        testing.assert_array_equal(x, y)

    @testing.multi_gpu(2)
    def test_array_multi_device_zero_size(self):
        with cuda.Device(0):
            x = testing.shaped_arange((0,), cupy, dtype='f')
        with cuda.Device(1):
            y = cupy.array(x)
        assert isinstance(y, cupy.ndarray)
        assert x is not y  # Do copy
        assert x.device.id == 0
        assert y.device.id == 1
        testing.assert_array_equal(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_no_copy_ndmin(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, ndmin=5)
        assert a.shape == (2, 3, 4)
        a.fill(0)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.asarray(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_is_not_copied(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a)
        a.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_preserves_numpy_array_order(self, xp, dtype, order):
        a_numpy = testing.shaped_arange((2, 3, 4), numpy, dtype, order)
        b = xp.asarray(a_numpy)
        assert b.flags.f_contiguous == a_numpy.flags.f_contiguous
        assert b.flags.c_contiguous == a_numpy.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asanyarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asanyarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_with_order_copy_behavior(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        a.fill(0)
        return b

    @testing.for_all_dtypes()
    def test_asarray_cuda_array_interface(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(DummyObjectWithCudaArrayInterface(a))
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_asarray_cuda_array_interface_is_not_copied(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(DummyObjectWithCudaArrayInterface(a))
        a.fill(0)
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_asarray_cuda_array_interface_order(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(DummyObjectWithCudaArrayInterface(a), order='F')
        assert b.flags.f_contiguous
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_asarray_cuda_array_interface_with_strides(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype).T
        b = cupy.asarray(DummyObjectWithCudaArrayInterface(a))
        assert a.strides == b.strides
        assert a.nbytes == b.data.mem.size

    # TODO(leofang): remove this test when masked array is supported
    def test_asarray_cuda_array_interface_with_masked_array(self):
        a = cupy.arange(10)
        mask = cupy.zeros(10)
        a = DummyObjectWithCudaArrayInterface(a, mask)
        with pytest.raises(ValueError) as ex:
            b = cupy.asarray(a)  # noqa
        assert 'does not support' in str(ex.value)

    def test_ascontiguousarray_on_noncontiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = a.transpose(2, 0, 1)
        c = cupy.ascontiguousarray(b)
        assert c.flags.c_contiguous
        testing.assert_array_equal(b, c)

    def test_ascontiguousarray_on_contiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = cupy.ascontiguousarray(a)
        assert a is b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copy(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = xp.copy(a, order=order)
        a[1] = 1
        return b

    @testing.multi_gpu(2)
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_copy_multigpu(self, dtype, order):
        with cuda.Device(0):
            src = cupy.random.uniform(-1, 1, (2, 3)).astype(dtype)
        with cuda.Device(1):
            dst = cupy.copy(src, order)
        testing.assert_allclose(src, dst, rtol=0, atol=0)

    @testing.for_CF_orders()
    @testing.numpy_cupy_equal()
    def test_copy_order(self, xp, order):
        a = xp.zeros((2, 3, 4), order=order)
        b = xp.copy(a)
        return (b.flags.c_contiguous, b.flags.f_contiguous)


class DummyObjectWithCudaArrayInterface(object):

    def __init__(self, a, mask=None):
        self.a = a
        self.mask = mask

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.a.shape,
            'strides': self.a.strides,
            'typestr': self.a.dtype.str,
            'descr': self.a.dtype.descr,
            'data': (self.a.data.ptr, False),
            'version': 2,
        }
        if self.mask is not None:
            desc['mask'] = self.mask
        return desc


@testing.parameterize(
    *testing.product({
        'ndmin': [0, 1, 2, 3],
        'copy': [True, False],
        'xp': [numpy, cupy]
    })
)
class TestArrayPreservationOfShape(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_cupy_array(self, dtype):
        shape = 2, 3
        a = testing.shaped_arange(shape, self.xp, dtype)
        cupy.array(a, copy=self.copy, ndmin=self.ndmin)

        # Check if cupy.ndarray does not alter
        # the shape of the original array.
        assert a.shape == shape


@testing.parameterize(
    *testing.product({
        'ndmin': [0, 1, 2, 3],
        'copy': [True, False],
        'xp': [numpy, cupy]
    })
)
class TestArrayCopy(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_cupy_array(self, dtype):
        a = testing.shaped_arange((2, 3), self.xp, dtype)
        actual = cupy.array(a, copy=self.copy, ndmin=self.ndmin)

        should_copy = (self.xp is numpy) or self.copy
        # TODO(Kenta Oono): Better determination of copy.
        is_copied = not ((actual is a) or (actual.base is a) or
                         (actual.base is a.base and a.base is not None))
        assert should_copy == is_copied


class TestArrayInvalidObject(unittest.TestCase):

    def test_invalid_type(self):
        a = numpy.array([1, 2, 3], dtype=object)
        with self.assertRaises(ValueError):
            cupy.array(a)
