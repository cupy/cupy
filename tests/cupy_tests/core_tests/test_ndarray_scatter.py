import unittest

import numpy

import cupy
from cupy import testing


@testing.parameterize(
    # array only
    {'shape': (2, 3, 4), 'slices': numpy.array(-1), 'value': 1},
    {'shape': (2, 3, 4), 'slices': numpy.array([1, 0]), 'value': 1},
    {'shape': (2, 3, 4), 'slices': (slice(None), [1, 2]), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (slice(None), [[1, 2], [0, -1]],), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (slice(None), slice(None), [[1, 2], [0, 3]]), 'value': 1},
    # slice and array
    {'shape': (3, 4, 5),
     'slices': (slice(None), slice(1, 2), [[1, 3], [0, 2]]), 'value': 1},
    # None and array
    {'shape': (3, 4, 5),
     'slices': (None, [1, -1]), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (None, [1, -1], None), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (None, None, None, [1, -1]), 'value': 1},
    # None, slice and array
    {'shape': (3, 4, 5),
     'slices': (slice(0, 1), None, [1, -1]), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (slice(0, 1), slice(1, 2), [1, -1]), 'value': 1},
    {'shape': (3, 4, 5),
     'slices': (slice(0, 1), None, slice(1, 2), [1, -1]), 'value': 1},
    # broadcasting
    {'shape': (3, 4, 5), 'slices': (slice(None), [[1, 2], [0, -1]],),
     'value': numpy.arange(3 * 2 * 2 * 5).reshape(3, 2, 2, 5)},
)
@testing.gpu
class TestScatterAddNoDuplicate(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32])
    @testing.numpy_cupy_array_equal()
    def test_scatter_add(self, xp, dtype):
        a = xp.zeros(self.shape, dtype)
        if xp is cupy:
            a.scatter_add(self.slices, self.value)
        else:
            a[self.slices] = a[self.slices] + self.value
        return a


@testing.parameterize(
    {'shape': (2, 3), 'slices': ([1, 1], slice(None)), 'value': 1,
     'expected': numpy.array([[0, 0, 0], [2, 2, 2]])},
    {'shape': (2, 3), 'slices': ([1, 0, 1], slice(None)), 'value': 1,
     'expected': numpy.array([[1, 1, 1], [2, 2, 2]])},
    {'shape': (2, 3), 'slices': (slice(1, 2), [1, 0, 1]), 'value': 1,
     'expected': numpy.array([[0, 0, 0], [1, 2, 0]])},
)
@testing.gpu
class TestScatterAddDuplicateVectorValue(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32])
    def test_scatter_add(self, dtype):
        a = cupy.zeros(self.shape, dtype)
        a.scatter_add(self.slices, self.value)

        numpy.testing.assert_almost_equal(a.get(), self.expected)


@testing.gpu
class TestScatterAddCupyArguments(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32])
    def test_scatter_add_cupy_arguments(self, dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype)
        slices = (cupy.array([1, 1]), slice(None))
        a.scatter_add(slices, cupy.array(1.))
        testing.assert_array_equal(
            a, cupy.array([[0., 0., 0.], [2., 2., 2.]], dtype))

    @testing.for_dtypes(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong], name='src_dtype')
    @testing.for_dtypes(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong], name='dst_dtype')
    def test_scatter_add_differnt_dtypes(self, src_dtype, dst_dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype=src_dtype)
        value = cupy.array(1, dtype=dst_dtype)
        slices = ([1, 1], slice(None))
        a.scatter_add(slices, value)

        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[0, 0, 0], [2, 2, 2]], dtype=src_dtype))
