import unittest

import cupy
from cupy.core import core
from cupy import testing
import numpy


class TestGetStridesForNocopyReshape(unittest.TestCase):

    def test_different_size(self):
        a = core.ndarray((2, 3))
        self.assertEqual(core._get_strides_for_nocopy_reshape(a, (1, 5)),
                         [])

    def test_one(self):
        a = core.ndarray((1,), dtype=cupy.int32)
        self.assertEqual(core._get_strides_for_nocopy_reshape(a, (1, 1, 1)),
                         [4, 4, 4])

    def test_normal(self):
        # TODO(nno): write test for normal case
        pass


@testing.parameterize(*[
    {'xp': numpy},
    {'xp': cupy}
])
class TestArrayDtype(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_cupy_ndarray(self, dtype):
        a = self.xp.random.randint(0, 3, (2, 3))
        b = cupy.array(a, dtype)
        self.assertEqual(dtype, b.dtype)
        a = a.astype(b.dtype)
        testing.assert_array_equal(a, b)

    def test_cupy_ndarray_dtype_is_none(self):
        a = self.xp.random.randint(0, 3, (2, 3))
        b = cupy.array(a, None)
        testing.assert_array_equal(a, b)


class TestArrayCopy(unittest.TestCase):

    def test_copy(self):
        a = cupy.random.randint(0, 3, (2, 3))
        b = cupy.array(a, copy=False)
        self.assertIs(a, b)


@testing.parameterize(
    *testing.product({
        'ndmin': [0, 1, 2, 3],
        'copy': [True, False],
        'xp': [numpy, cupy]
    })
)
class TestArrayNdmin(unittest.TestCase):

    def test_cupy_ndarray_ndmin(self):
        shape = 2, 3
        a = self.xp.arange(6).reshape(*shape)
        ndim = a.ndim
        actual = cupy.array(a, copy=self.copy, ndmin=self.ndmin)

        # Check if cupy.ndarray does not alter
        # the shape of the original array.
        self.assertEqual(a.shape, shape)

        expected = a
        if ndim < self.ndmin:
            expected_shape = (1,) * (self.ndmin - ndim) + a.shape
            expected = a.reshape(expected_shape)
        testing.assert_array_equal(expected, actual)

        if (self.xp is cupy) and not self.copy:
            self.assertTrue((a is actual) or (actual.base is a.base))


class TestArrayInvalidObject(unittest.TestCase):

    def test_invalid_type(self):
        a = numpy.array([1, 2, 3], dtype=object)
        with self.assertRaises(ValueError):
            cupy.array(a)
