import copy
import unittest

import numpy

import cupy
from cupy import core
from cupy import cuda
from cupy import get_array_module
from cupy import testing


class TestGetSize(unittest.TestCase):

    def test_none(self):
        self.assertEqual(core.get_size(None), ())

    def check_collection(self, a):
        self.assertEqual(core.get_size(a), tuple(a))

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        self.assertEqual(core.get_size(1), (1,))

    def test_float(self):
        with self.assertRaises(ValueError):
            core.get_size(1.0)


def wrap_take(array, *args, **kwargs):
    if get_array_module(array) == numpy:
        kwargs["mode"] = "wrap"

    return array.take(*args, **kwargs)


@testing.parameterize(
    {'arg': None, 'shape': ()},
    {'arg': 3, 'shape': (3,)},
)
@testing.gpu
class TestNdarrayInit(unittest.TestCase):

    def test_shape(self):
        a = core.ndarray(self.arg)
        self.assertTupleEqual(a.shape, self.shape)


@testing.gpu
class TestNdarrayOrder(unittest.TestCase):

    shape = (2, 3, 4)

    def test_order(self):
        a = core.ndarray(self.shape, order='F')
        a_cpu = numpy.ndarray(self.shape, order='F')
        self.assertTupleEqual(a.strides, a_cpu.strides)
        self.assertTrue(a.flags.f_contiguous)
        self.assertTrue(not a.flags.c_contiguous)

    def test_order_none(self):
        a = core.ndarray(self.shape, order=None)
        a_cpu = numpy.ndarray(self.shape, order=None)
        self.assertEqual(a.flags.c_contiguous, a_cpu.flags.c_contiguous)
        self.assertEqual(a.flags.f_contiguous, a_cpu.flags.f_contiguous)
        self.assertTupleEqual(a.strides, a_cpu.strides)


@testing.gpu
class TestNdarrayInitRaise(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2, 3), dtype=object)
        with self.assertRaises(ValueError):
            core.array(arr)


@testing.parameterize(
    *testing.product({
        'shape': [(), (0,), (1,), (0, 0, 2), (2, 3)],
    })
)
@testing.gpu
class TestNdarrayCopy(unittest.TestCase):

    def _check_deepcopy(self, arr, arr2):
        self.assertIsNot(arr.data, arr2.data)
        self.assertEqual(arr.shape, arr2.shape)
        self.assertEqual(arr.size, arr2.size)
        self.assertEqual(arr.dtype, arr2.dtype)
        self.assertEqual(arr.strides, arr2.strides)
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
        self.assertEqual(arr2.device, arr.device)


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
        self.assertEqual(set(iface.keys()),
                         set(['shape', 'typestr', 'data', 'version', 'descr']))
        self.assertEqual(iface['shape'], (2, 3))
        self.assertEqual(iface['typestr'], '<f8')
        self.assertIsInstance(iface['data'], tuple)
        self.assertEqual(len(iface['data']), 2)
        self.assertEqual(iface['data'][0], arr.data.ptr)
        self.assertEqual(iface['data'][1], False)
        self.assertEqual(iface['version'], 0)
        self.assertEqual(iface['descr'], [('', '<f8')])

    def test_cuda_array_interface_view(self):
        arr = cupy.zeros(shape=(10, 20), dtype=cupy.float64)
        view = arr[::2, ::5]
        iface = view.__cuda_array_interface__
        self.assertEqual(set(iface.keys()),
                         set(['shape', 'typestr', 'data', 'version',
                              'strides', 'descr']))
        self.assertEqual(iface['shape'], (5, 4))
        self.assertEqual(iface['typestr'], '<f8')
        self.assertIsInstance(iface['data'], tuple)
        self.assertEqual(len(iface['data']), 2)
        self.assertEqual(iface['data'][0], arr.data.ptr)
        self.assertEqual(iface['data'][1], False)
        self.assertEqual(iface['version'], 0)
        self.assertEqual(iface['strides'], [320, 40])
        self.assertEqual(iface['descr'], [('', '<f8')])


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
    {"shape": (3, 4, 5), "indices": (2,), "axis": 3},
    {"shape": (), "indices": (0,), "axis": 2}
)
@testing.gpu
class TestNdarrayTakeErrorAxisOverRun(unittest.TestCase):

    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_raises()
    def test_axis_overrun1(self, xp):
        a = testing.shaped_arange(self.shape, xp)
        wrap_take(a, self.indices, axis=self.axis)

    def test_axis_overrun2(self):
        a = testing.shaped_arange(self.shape, cupy)
        with self.assertRaises(core.core._AxisError):
            wrap_take(a, self.indices, axis=self.axis)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 4)},
    {"shape": (), "indices": 0, "out_shape": (1,)}
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
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 3)},
    {"shape": (), "indices": 0, "out_shape": ()}
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
    {"shape": (0,), "indices": (0,)},
    {"shape": (0,), "indices": (0, 1)},
)
@testing.gpu
class TestZeroSizedNdarrayTake(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32)
        return wrap_take(a, i)


@testing.parameterize(
    {"shape": (0,), "indices": (1,)},
    {"shape": (0,), "indices": (1, 1)},
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
