import unittest

import numpy

from cupy import core
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


@testing.gpu
class TestNdarrayInitRaise(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2, 3), dtype=object)
        with self.assertRaises(ValueError):
            core.array(arr)


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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_axis_overrun(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        wrap_take(a, self.indices, axis=self.axis)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 4)},
    {"shape": (), "indices": 0, "out_shape": (1,)}
)
@testing.gpu
class TestNdarrayTakeErrorShapeMismatch(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        o = testing.shaped_arange(self.out_shape, xp, dtype)
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
