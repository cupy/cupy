import unittest
import pytest

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
    # array with duplicate indices
    {'shape': (2, 3), 'slices': ([1, 1], slice(None)), 'value': 1},
    {'shape': (2, 3), 'slices': ([1, 0, 1], slice(None)), 'value': 1},
    {'shape': (2, 3), 'slices': (slice(1, 2), [1, 0, 1]), 'value': 1},
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
    # multiple integer arrays
    {'shape': (2, 3, 4), 'slices': ([1, 0], [2, 1]),
     'value': numpy.arange(2 * 4).reshape(2, 4)},
    {'shape': (2, 3, 4), 'slices': ([1, 0], slice(None), [2, 1]),
     'value': numpy.arange(2 * 3).reshape(2, 3)},
    {'shape': (2, 3, 4), 'slices': ([1, 0], slice(None), [[2, 0], [3, 1]]),
     'value': numpy.arange(2 * 2 * 3).reshape(2, 2, 3)},
    {'shape': (1, 1, 2, 3, 4),
     'slices': (None, slice(None), 0, [1, 0], slice(0, 2, 2), [2, -1]),
     'value': 1},
    # multiple integer arrays duplicate
    {'shape': (2, 3, 4), 'slices': ([1, 1], [1, 1]),
     'value': numpy.arange(2 * 4).reshape(2, 4)},
    {'shape': (2, 3, 4), 'slices': ([1, 1], slice(None), [[2, 2], [3, 1]]),
     'value': numpy.arange(2 * 2 * 3).reshape(2, 2, 3)},
    {'shape': (2, 3, 4), 'slices': ([1, 1], 1, [[2, 2], [3, 1]]),
     'value': numpy.arange(2 * 2).reshape(2, 2)},
    # mask
    {'shape': (3, 4, 5),
     'slices': (numpy.random.choice([False, True], (3, 4, 5)),),
     'value': 1},
    {'shape': (3, 4, 5),
     'slices': (numpy.random.choice([False, True], (3,)),),
     'value': numpy.arange(4 * 5).reshape(4, 5)},
    {'shape': (3, 4, 5),
     'slices': (slice(None), numpy.array([True, False, False, True]),),
     'value': numpy.arange(3 * 2 * 5).reshape(3, 2, 5)},
    # empty arrays
    {'shape': (2, 3, 4), 'slices': [], 'value': 1},
    {'shape': (2, 3, 4), 'slices': [],
     'value': numpy.array([1, 1, 1, 1])},
    {'shape': (2, 3, 4), 'slices': [],
     'value': numpy.random.uniform(size=(3, 4))},
    {'shape': (2, 3, 4), 'slices': numpy.array([], dtype=numpy.int32),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': ([],),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': numpy.array([[]], dtype=numpy.int32),
     'value': numpy.random.uniform(size=(3, 4))},
    {'shape': (2, 3, 4), 'slices': ([[]],),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': ([[[]]],),
     'value': 1},
    {'shape': (2, 3, 4, 5), 'slices': ([[[]]],),
     'value': 1},
    {'shape': (2, 3, 4, 5), 'slices': ([[[[]]]],),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': (slice(None), []),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': ([], []),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': numpy.array([], dtype=numpy.bool_),
     'value': 1},
    {'shape': (2, 3, 4),
     'slices': (slice(None), numpy.array([], dtype=numpy.bool_)),
     'value': 1},
    {'shape': (2, 3, 4), 'slices': numpy.array([[], []], dtype=numpy.bool_),
     'value': numpy.random.uniform(size=(4,))},
    # list indexes
    {'shape': (2, 3, 4), 'slices': [1], 'value': 1},
    {'shape': (2, 3, 4), 'slices': [1, 1],
     'value': numpy.arange(2 * 3 * 4).reshape(2, 3, 4)},
    {'shape': (2, 3, 4), 'slices': ([1],), 'value': 1},
    {'shape': (2, 3, 4), 'slices': ([1, 1],), 'value': 1},
    {'shape': (2, 3, 4), 'slices': ([1], [1]), 'value': 1},
    {'shape': (2, 3, 4), 'slices': ([1, 1], 1), 'value': 1},
    {'shape': (2, 3, 4), 'slices': ([1], slice(1, 2)), 'value': 1},
    {'shape': (2, 3, 4), 'slices': ([[1]], slice(1, 2)), 'value': 1},
)
@testing.gpu
class TestScatterParametrized(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float16,
                         numpy.float64])
    @testing.numpy_cupy_array_equal()
    def test_scatter_add(self, xp, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        a = xp.zeros(self.shape, dtype)
        if xp is cupy:
            a.scatter_add(self.slices, self.value)
        else:
            numpy.add.at(a, self.slices, self.value)
        return a

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float64])
    @testing.numpy_cupy_array_equal()
    def test_scatter_max(self, xp, dtype):
        a = xp.zeros(self.shape, dtype)
        if xp is cupy:
            a.scatter_max(self.slices, self.value)
        else:
            numpy.maximum.at(a, self.slices, self.value)
        return a

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float64])
    @testing.numpy_cupy_array_equal()
    def test_scatter_min(self, xp, dtype):
        a = xp.zeros(self.shape, dtype)
        if xp is cupy:
            a.scatter_min(self.slices, self.value)
        else:
            numpy.minimum.at(a, self.slices, self.value)
        return a


@testing.gpu
class TestScatterAdd(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float16,
                         numpy.float64])
    def test_scatter_add_cupy_arguments(self, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (2, 3)
        a = cupy.zeros(shape, dtype)
        slices = (cupy.array([1, 1]), slice(None))
        a.scatter_add(slices, cupy.array(1.))
        testing.assert_array_equal(
            a, cupy.array([[0., 0., 0.], [2., 2., 2.]], dtype))

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float16,
                         numpy.float64])
    def test_scatter_add_cupy_arguments_mask(self, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (2, 3)
        a = cupy.zeros(shape, dtype)
        slices = (cupy.array([True, False]), slice(None))
        a.scatter_add(slices, cupy.array(1.))
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 1.], [0., 0., 0.]], dtype))

    @testing.for_dtypes_combination(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong, numpy.float16, numpy.float64],
        names=['src_dtype', 'dst_dtype'])
    def test_scatter_add_differnt_dtypes(self, src_dtype, dst_dtype):
        if (
                cupy.cuda.runtime.is_hip
                and (src_dtype == numpy.float16
                     or dst_dtype == numpy.float16)):
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (2, 3)
        a = cupy.zeros(shape, dtype=src_dtype)
        value = cupy.array(1, dtype=dst_dtype)
        slices = ([1, 1], slice(None))
        a.scatter_add(slices, value)

        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[0, 0, 0], [2, 2, 2]], dtype=src_dtype))

    @testing.for_dtypes_combination(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong, numpy.float16, numpy.float64],
        names=['src_dtype', 'dst_dtype'])
    def test_scatter_add_differnt_dtypes_mask(self, src_dtype, dst_dtype):
        if (
                cupy.cuda.runtime.is_hip
                and (src_dtype == numpy.float16
                     or dst_dtype == numpy.float16)):
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (2, 3)
        a = cupy.zeros(shape, dtype=src_dtype)
        value = cupy.array(1, dtype=dst_dtype)
        slices = (numpy.array([[True, False, False], [False, True, True]]))
        a.scatter_add(slices, value)

        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[1, 0, 0], [0, 1, 1]], dtype=src_dtype))


class TestScatterMinMax(unittest.TestCase):

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float64])
    def test_scatter_minmax_cupy_arguments(self, dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype)
        slices = (cupy.array([1, 1]), slice(None))
        a.scatter_max(slices, cupy.array(1.))
        testing.assert_array_equal(
            a, cupy.array([[0., 0., 0.], [1., 1., 1.]], dtype))

        a = cupy.ones(shape, dtype)
        a.scatter_min(slices, cupy.array(0.))
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 1.], [0., 0., 0.]], dtype))

    @testing.for_dtypes([numpy.float32, numpy.int32, numpy.uint32,
                         numpy.uint64, numpy.ulonglong, numpy.float64])
    def test_scatter_minmax_cupy_arguments_mask(self, dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype)
        slices = (cupy.array([True, False]), slice(None))
        a.scatter_max(slices, cupy.array(1.))
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 1.], [0., 0., 0.]], dtype))

        a = cupy.ones(shape, dtype)
        a.scatter_min(slices, cupy.array(0.))
        testing.assert_array_equal(
            a, cupy.array([[0., 0., 0.], [1., 1., 1.]], dtype))

    @testing.for_dtypes_combination(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong, numpy.float64],
        names=['src_dtype', 'dst_dtype'])
    def test_scatter_minmax_differnt_dtypes(self, src_dtype, dst_dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype=src_dtype)
        value = cupy.array(1, dtype=dst_dtype)
        slices = ([1, 1], slice(None))
        a.scatter_max(slices, value)
        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[0, 0, 0], [1, 1, 1]], dtype=src_dtype))

        a = cupy.ones(shape, dtype=src_dtype)
        value = cupy.array(0, dtype=dst_dtype)
        a.scatter_min(slices, value)
        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[1, 1, 1], [0, 0, 0]], dtype=src_dtype))

    @testing.for_dtypes_combination(
        [numpy.float32, numpy.int32, numpy.uint32, numpy.uint64,
         numpy.ulonglong, numpy.float16, numpy.float64],
        names=['src_dtype', 'dst_dtype'])
    def test_scatter_minmax_differnt_dtypes_mask(self, src_dtype, dst_dtype):
        shape = (2, 3)
        a = cupy.zeros(shape, dtype=src_dtype)
        value = cupy.array(1, dtype=dst_dtype)
        slices = (numpy.array([[True, False, False], [False, True, True]]))
        a.scatter_max(slices, value)
        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[1, 0, 0], [0, 1, 1]], dtype=src_dtype))

        a = cupy.ones(shape, dtype=src_dtype)
        value = cupy.array(0, dtype=dst_dtype)
        a.scatter_min(slices, value)
        numpy.testing.assert_almost_equal(
            a.get(),
            numpy.array([[0, 1, 1], [1, 0, 0]], dtype=src_dtype))
