import unittest

import numpy

from cupy import core
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


@testing.parameterize(
    {'arg': None, 'shape': ()},
    {'arg': 3, 'shape': (3,)},
)
class TestNdarrayInit(unittest.TestCase):

    def test_shape(self):
        a = core.ndarray(self.arg)
        self.assertTupleEqual(a.shape, self.shape)


class TestNdarrayInitRaise(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2, 3), dtype=object)
        with self.assertRaises(ValueError):
            core.array(arr)


@testing.parameterize(
    *testing.product({
        'indices_shape': [(2,), (2, 3)],
        'axis': [None, 0, 1, 2, -1, -2],
    })
)
class TestNdarrayTake(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=False)
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.axis is None:
            m = a.size
        else:
            m = a.shape[self.axis]
        i = testing.shaped_arange(self.indices_shape, xp, numpy.int32) % m
        return a.take(i, self.axis)


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2]],
        'axis': [None, 0, 1, -1, -2],
    })
)
class TestNdarrayTakeWithInt(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=False)
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a.take(self.indices, self.axis)


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2]],
        'axis': [None, 0, 1, -1, -2],
    })
)
class TestNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=False)
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = a.take(self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        a.take(self.indices, self.axis, out=r2)
        assert (r1 == r2).all()
        return r2


@testing.parameterize(
    *testing.product({
        'indices': [0, -1, [0], [0, -1]],
        'axis': [None, 0, -1],
    })
)
class TestScalaNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = ()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=False)
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = a.take(self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        a.take(self.indices, self.axis, out=r2)
        assert (r1 == r2).all()
        return r2


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2,), "axis": 3},
    {"shape": (), "indices": (0,), "axis": 2}
)
class TestNdarrayTakeErrorAxisOverRun(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_axis_overrun(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        a.take(self.indices, axis=self.axis)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 4)},
    {"shape": (), "indices": 0, "out_shape": (1,)}
)
class TestNdarrayTakeErrorShapeMismatch(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        o = testing.shaped_arange(self.out_shape, xp, dtype)
        a.take(i, out=o)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 3)},
    {"shape": (), "indices": 0, "out_shape": ()}
)
class TestNdarrayTakeErrorTypeMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        o = testing.shaped_arange(self.out_shape, xp, numpy.float32)
        a.take(i, out=o)
