import copy
import unittest

import numpy
import pytest

import cupy
from cupy import core
from cupy import cuda
from cupy import get_array_module
from cupy import testing


class TestGetSize(unittest.TestCase):

    def test_none(self):
        assert core.get_size(None) == ()

    def check_collection(self, a):
        assert core.get_size(a) == tuple(a)

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        assert core.get_size(1) == (1,)

    def test_float(self):
        with pytest.raises(ValueError):
            core.get_size(1.0)


def wrap_take(array, *args, **kwargs):
    if get_array_module(array) == numpy:
        kwargs['mode'] = 'wrap'

    return array.take(*args, **kwargs)


@testing.gpu
class TestNdarrayInit(unittest.TestCase):

    def test_shape_none(self):
        a = cupy.ndarray(None)
        assert a.shape == ()

    def test_shape_int(self):
        a = cupy.ndarray(3)
        assert a.shape == (3,)

    def test_shape_int_with_strides(self):
        dummy = cupy.ndarray(3)
        a = cupy.ndarray(3, strides=(0,), memptr=dummy.data)
        assert a.shape == (3,)
        assert a.strides == (0,)

    def test_memptr(self):
        a = cupy.arange(6).astype(numpy.float32).reshape((2, 3))
        memptr = a.data

        b = cupy.ndarray((2, 3), numpy.float32, memptr)
        testing.assert_array_equal(a, b)

        b += 1
        testing.assert_array_equal(a, b)

    def test_memptr_with_strides(self):
        buf = cupy.ndarray(20, numpy.uint8)
        memptr = buf.data

        # self-overlapping strides
        a = cupy.ndarray((2, 3), numpy.float32, memptr, strides=(8, 4))
        assert a.strides == (8, 4)

        a[:] = 1
        a[0, 2] = 4
        assert float(a[1, 0]) == 4

    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_strides_without_memptr(self, xp):
        xp.ndarray((2, 3), numpy.float32, strides=(20, 4))

    def test_strides_is_given_and_order_is_ignored(self):
        buf = cupy.ndarray(20, numpy.uint8)
        a = cupy.ndarray(
            (2, 3), numpy.float32, buf.data, strides=(8, 4), order='C')
        assert a.strides == (8, 4)

    @testing.numpy_cupy_raises(accept_error=TypeError)
    def test_strides_is_given_but_order_is_invalid(self, xp):
        xp.ndarray((2, 3), numpy.float32, strides=(8, 4), order='!')

    def test_order(self):
        shape = (2, 3, 4)
        a = core.ndarray(shape, order='F')
        a_cpu = numpy.ndarray(shape, order='F')
        assert a.strides == a_cpu.strides
        assert a.flags.f_contiguous
        assert not a.flags.c_contiguous

    def test_order_none(self):
        shape = (2, 3, 4)
        a = core.ndarray(shape, order=None)
        a_cpu = numpy.ndarray(shape, order=None)
        assert a.flags.c_contiguous == a_cpu.flags.c_contiguous
        assert a.flags.f_contiguous == a_cpu.flags.f_contiguous
        assert a.strides == a_cpu.strides


@testing.parameterize(
    *testing.product({
        'shape': [(), (1,), (1, 2), (1, 2, 3)],
        'order': ['C', 'F'],
        'dtype': [
            numpy.uint8,  # itemsize=1
            numpy.uint16,  # itemsize=2
        ],
    }))
@testing.gpu
class TestNdarrayInitStrides(unittest.TestCase):

    # Check the strides given shape, itemsize and order.

    @testing.numpy_cupy_equal()
    def test_strides(self, xp):
        arr = xp.ndarray(self.shape, dtype=self.dtype, order=self.order)
        return (
            arr.strides,
            arr.flags.c_contiguous,
            arr.flags.f_contiguous)


@testing.gpu
class TestNdarrayInitRaise(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2, 3), dtype=object)
        with pytest.raises(ValueError):
            core.array(arr)


@testing.parameterize(
    *testing.product({
        'shape': [(), (0,), (1,), (0, 0, 2), (2, 3)],
    })
)
@testing.gpu
class TestNdarrayDeepCopy(unittest.TestCase):

    def _check_deepcopy(self, arr, arr2):
        assert arr.data is not arr2.data
        assert arr.shape == arr2.shape
        assert arr.size == arr2.size
        assert arr.dtype == arr2.dtype
        assert arr.strides == arr2.strides
        testing.assert_array_equal(arr, arr2)

    def test_deepcopy(self):
        arr = core.ndarray(self.shape)
        arr2 = copy.deepcopy(arr)
        self._check_deepcopy(arr, arr2)

    @testing.multi_gpu(2)
    def test_deepcopy_multi_device(self):
        arr = core.ndarray(self.shape)
        with cuda.Device(1):
            arr2 = copy.deepcopy(arr)
        self._check_deepcopy(arr, arr2)
        assert arr2.device == arr.device


@testing.gpu
class TestNdarrayCopy(unittest.TestCase):

    @testing.multi_gpu(2)
    @testing.for_orders('CFA')
    def test_copy_multi_device_non_contiguous(self, order):
        arr = core.ndarray((20,))[::2]
        dev1 = cuda.Device(1)
        with dev1:
            arr2 = arr.copy(order)
        assert arr2.device == dev1
        testing.assert_array_equal(arr, arr2)

    @testing.multi_gpu(2)
    def test_copy_multi_device_non_contiguous_K(self):
        arr = core.ndarray((20,))[::2]
        with cuda.Device(1):
            with self.assertRaises(NotImplementedError):
                arr.copy('K')


@testing.gpu
class TestNdarrayShape(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_shape_set(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = (3, 2)
        return xp.array(arr.shape)

    @testing.numpy_cupy_array_equal()
    def test_shape_set_infer(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = (3, -1)
        return xp.array(arr.shape)

    @testing.numpy_cupy_array_equal()
    def test_shape_set_int(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = 6
        return xp.array(arr.shape)


class TestNdarrayCudaInterface(unittest.TestCase):

    def test_cuda_array_interface(self):
        arr = cupy.zeros(shape=(2, 3), dtype=cupy.float64)
        iface = arr.__cuda_array_interface__
        assert (set(iface.keys()) ==
                set(['shape', 'typestr', 'data', 'version', 'descr',
                     'strides']))
        assert iface['shape'] == (2, 3)
        assert iface['typestr'] == '<f8'
        assert isinstance(iface['data'], tuple)
        assert len(iface['data']) == 2
        assert iface['data'][0] == arr.data.ptr
        assert not iface['data'][1]
        assert iface['version'] == 2
        assert iface['descr'] == [('', '<f8')]
        assert iface['strides'] is None

    def test_cuda_array_interface_view(self):
        arr = cupy.zeros(shape=(10, 20), dtype=cupy.float64)
        view = arr[::2, ::5]
        iface = view.__cuda_array_interface__
        assert (set(iface.keys()) ==
                set(['shape', 'typestr', 'data', 'version',
                     'strides', 'descr']))
        assert iface['shape'] == (5, 4)
        assert iface['typestr'] == '<f8'
        assert isinstance(iface['data'], tuple)
        assert len(iface['data']) == 2
        assert iface['data'][0] == arr.data.ptr
        assert not iface['data'][1]
        assert iface['version'] == 2
        assert iface['strides'] == (320, 40)
        assert iface['descr'] == [('', '<f8')]

    def test_cuda_array_interface_zero_size(self):
        arr = cupy.zeros(shape=(10,), dtype=cupy.float64)
        view = arr[0:3:-1]
        iface = view.__cuda_array_interface__
        assert (set(iface.keys()) ==
                set(['shape', 'typestr', 'data', 'version',
                     'strides', 'descr']))
        assert iface['shape'] == (0,)
        assert iface['typestr'] == '<f8'
        assert isinstance(iface['data'], tuple)
        assert len(iface['data']) == 2
        assert iface['data'][0] == 0
        assert not iface['data'][1]
        assert iface['version'] == 2
        assert iface['strides'] is None
        assert iface['descr'] == [('', '<f8')]


@testing.parameterize(
    *testing.product({
        'indices_shape': [(2,), (2, 3)],
        'axis': [None, 0, 1, 2, -1, -2],
    })
)
@testing.gpu
class TestNdarrayTake(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.axis is None:
            m = a.size
        else:
            m = a.shape[self.axis]
        i = testing.shaped_arange(self.indices_shape, xp, numpy.int32) % m
        return wrap_take(a, i, self.axis)


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2]],
        'axis': [None, 0, 1, -1, -2],
    })
)
@testing.gpu
class TestNdarrayTakeWithInt(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return wrap_take(a, self.indices, self.axis)


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2]],
        'axis': [None, 0, 1, -1, -2],
    })
)
@testing.gpu
class TestNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = wrap_take(a, self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        wrap_take(a, self.indices, self.axis, out=r2)
        testing.assert_array_equal(r1, r2)
        return r2


@testing.parameterize(
    *testing.product({
        'indices': [0, -1, [0], [0, -1]],
        'axis': [None, 0, -1],
    })
)
@testing.gpu
class TestScalaNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = ()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = wrap_take(a, self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        wrap_take(a, self.indices, self.axis, out=r2)
        testing.assert_array_equal(r1, r2)
        return r2


@testing.parameterize(
    {'shape': (3, 4, 5), 'indices': (2,), 'axis': 3},
    {'shape': (), 'indices': (0,), 'axis': 2}
)
@testing.gpu
class TestNdarrayTakeErrorAxisOverRun(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_axis_overrun1(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        wrap_take(a, self.indices, axis=self.axis)

    def test_axis_overrun2(self):
        a = testing.shaped_arange(self.shape, cupy)
        with pytest.raises(numpy.AxisError):
            wrap_take(a, self.indices, axis=self.axis)


@testing.parameterize(
    {'shape': (3, 4, 5), 'indices': (2, 3), 'out_shape': (2, 4)},
    {'shape': (), 'indices': 0, 'out_shape': (1,)}
)
@testing.gpu
class TestNdarrayTakeErrorShapeMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        o = testing.shaped_arange(self.out_shape, xp)
        wrap_take(a, i, out=o)


@testing.parameterize(
    {'shape': (3, 4, 5), 'indices': (2, 3), 'out_shape': (2, 3)},
    {'shape': (), 'indices': 0, 'out_shape': ()}
)
@testing.gpu
class TestNdarrayTakeErrorTypeMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        o = testing.shaped_arange(self.out_shape, xp, numpy.float32)
        wrap_take(a, i, out=o)


@testing.parameterize(
    {'shape': (0,), 'indices': (0,)},
    {'shape': (0,), 'indices': (0, 1)},
)
@testing.gpu
class TestZeroSizedNdarrayTake(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32)
        return wrap_take(a, i)


@testing.parameterize(
    {'shape': (0,), 'indices': (1,)},
    {'shape': (0,), 'indices': (1, 1)},
)
@testing.gpu
class TestZeroSizedNdarrayTakeIndexError(unittest.TestCase):

    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32)
        wrap_take(a, i)


@testing.gpu
class TestSize(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_size_without_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x)

    @testing.numpy_cupy_equal()
    def test_size_with_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, 0)

    @testing.numpy_cupy_equal()
    def test_size_with_negative_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, -1)

    @testing.numpy_cupy_equal()
    def test_size_zero_dim_array(self, xp):
        x = testing.shaped_arange((), xp, numpy.int32)
        return xp.size(x)

    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_size_axis_too_large(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, 3)

    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_size_axis_too_small(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, -4)

    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_size_zero_dim_array_with_axis(self, xp):
        x = testing.shaped_arange((), xp, numpy.int32)
        return xp.size(x, 0)


@testing.gpu
class TestPythonInterface(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 5), xp, dtype)
        return bytes(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_empty(self, xp, dtype):
        x = xp.empty((3, 4, 5), dtype)
        return bytes(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_empty2(self, xp, dtype):
        x = xp.empty((), dtype)
        return bytes(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_scalar(self, xp, dtype):
        x = xp.array([3], dtype).item()
        return bytes(x)
