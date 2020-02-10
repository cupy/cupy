import copy
import unittest

import numpy

import cupy
from cupy import testing


@testing.parameterize(
    *testing.product({
        'assertion': ['assert_allclose', 'assert_array_almost_equal',
                      'assert_array_almost_equal_nulp',
                      'assert_array_max_ulp', 'assert_array_equal'],
        'array_module_x': [numpy, cupy],
        'array_module_y': [numpy, cupy]
    })
)
@testing.gpu
class TestEqualityAssertion(unittest.TestCase):

    def setUp(self):
        self.assertion = getattr(testing, self.assertion)
        val = numpy.random.uniform(-1, 1, (2, 3))
        self.x = self.array_module_x.array(val, val.dtype, copy=True)
        self.y = self.array_module_y.array(val, val.dtype, copy=True)

    def test_equality(self):
        self.assertion(self.x, self.y)

    def test_inequality(self):
        self.y += 1
        with self.assertRaises(AssertionError):
            self.assertion(self.x, self.y)


def _convert_array(xs, array_module):
    if array_module == 'all_numpy':
        return xs
    elif array_module == 'all_cupy':
        return [
            cupy.asarray(x)
            for x in xs
        ]
    else:
        return [
            cupy.asarray(x) if numpy.random.randint(0, 2) else x
            for x in xs
        ]


@testing.parameterize(
    *testing.product({
        'array_module_x': ['all_numpy', 'all_cupy', 'random'],
        'array_module_y': ['all_numpy', 'all_cupy', 'random']
    })
)
@testing.gpu
class TestListEqualityAssertion(unittest.TestCase):

    def setUp(self):
        xs = [numpy.random.uniform(-1, 1, (2, 3)) for _ in range(10)]
        ys = copy.deepcopy(xs)
        self.xs = _convert_array(xs, self.array_module_x)
        self.ys = _convert_array(ys, self.array_module_y)

    def test_equality_numpy(self):
        testing.assert_array_list_equal(self.xs, self.ys)

    def test_inequality_numpy(self):
        self.xs[0] += 1
        with self.assertRaisesRegex(
                AssertionError, '^\nArrays are not equal'):
            testing.assert_array_list_equal(self.xs, self.ys)


@testing.parameterize(
    *testing.product({
        'array_module_x': [numpy, cupy],
        'array_module_y': [numpy, cupy]
    })
)
@testing.gpu
class TestStridesEqualityAssertion(unittest.TestCase):

    def setUp(self):
        val = numpy.random.uniform(-1, 1, (2, 3))
        self.x = self.array_module_x.array(val, val.dtype, copy=True)
        self.y = self.array_module_y.array(val, val.dtype, copy=True)

    def test_equality_numpy(self):
        testing.assert_array_equal(self.x, self.y, strides_check=True)

    def test_inequality_numpy(self):
        self.y = self.array_module_y.asfortranarray(self.y)
        with self.assertRaises(AssertionError):
            testing.assert_array_equal(self.x, self.y, strides_check=True)


@testing.parameterize(
    *testing.product({
        'array_module_x': [numpy, cupy],
        'array_module_y': [numpy, cupy]
    })
)
@testing.gpu
class TestLessAssertion(unittest.TestCase):

    def setUp(self):
        val = numpy.random.uniform(-1, 1, (2, 3))
        self.x = self.array_module_x.array(val, val.dtype, copy=True)
        self.y = self.array_module_y.array(val + 1, val.dtype, copy=True)

    def test_equality_numpy(self):
        testing.assert_array_less(self.x, self.y)

    def test_inequality_numpy(self):
        self.x[0] += 100
        with self.assertRaises(AssertionError):
            testing.assert_array_less(self.x, self.y)
