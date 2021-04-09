import math
import unittest

import pytest

from cupy._core import internal
from cupy import testing


class TestProd(unittest.TestCase):

    def test_empty(self):
        assert internal.prod([]) == 1

    def test_one(self):
        assert internal.prod([2]) == 2

    def test_two(self):
        assert internal.prod([2, 3]) == 6


class TestProdSequence(unittest.TestCase):

    def test_empty(self):
        assert internal.prod_sequence(()) == 1

    def test_one(self):
        assert internal.prod_sequence((2,)) == 2

    def test_two(self):
        assert internal.prod_sequence((2, 3)) == 6


class TestGetSize(unittest.TestCase):

    def test_none(self):
        with testing.assert_warns(DeprecationWarning):
            assert internal.get_size(None) == ()

    def check_collection(self, a):
        assert internal.get_size(a) == tuple(a)

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        assert internal.get_size(1) == (1,)

    def test_float(self):
        with pytest.raises(ValueError):
            internal.get_size(1.0)


class TestVectorEqual(unittest.TestCase):

    def test_empty(self):
        assert internal.vector_equal([], []) is True

    def test_not_equal(self):
        assert internal.vector_equal([1, 2, 3], [1, 2, 0]) is False

    def test_equal(self):
        assert internal.vector_equal([-1, 0, 1], [-1, 0, 1]) is True

    def test_different_size(self):
        assert internal.vector_equal([1, 2, 3], [1, 2]) is False


class TestGetCContiguity(unittest.TestCase):

    def test_zero_in_shape(self):
        assert internal.get_c_contiguity((1, 0, 1), (1, 1, 1), 3)

    def test_all_one_shape(self):
        assert internal.get_c_contiguity((1, 1, 1), (1, 1, 1), 3)

    def test_normal1(self):
        assert internal.get_c_contiguity((3, 4, 3), (24, 6, 2), 2)

    def test_normal2(self):
        assert internal.get_c_contiguity((3, 1, 3), (6, 100, 2), 2)

    def test_normal3(self):
        assert internal.get_c_contiguity((3,), (4, ), 4)

    def test_normal4(self):
        assert internal.get_c_contiguity((), (), 4)

    def test_normal5(self):
        assert internal.get_c_contiguity((3, 1), (4, 8), 4)

    def test_no_contiguous1(self):
        assert not internal.get_c_contiguity((3, 4, 3), (30, 6, 2), 2)

    def test_no_contiguous2(self):
        assert not internal.get_c_contiguity((3, 1, 3), (24, 6, 2), 2)

    def test_no_contiguous3(self):
        assert not internal.get_c_contiguity((3, 1, 3), (6, 6, 4), 2)


class TestInferUnknownDimension(unittest.TestCase):

    def test_known_all(self):
        assert internal.infer_unknown_dimension((1, 2, 3), 6) == [1, 2, 3]

    def test_multiple_unknown(self):
        with self.assertRaises(ValueError):
            internal.infer_unknown_dimension((-1, 1, -1), 10)

    def test_infer(self):
        assert internal.infer_unknown_dimension((-1, 2, 3), 12) == [2, 2, 3]


@testing.parameterize(
    {'slice': (2, 8, 1),    'expect': (2, 8, 1)},
    {'slice': (2, None, 1), 'expect': (2, 10, 1)},
    {'slice': (2, 1, 1),    'expect': (2, 2, 1)},
    {'slice': (2, -1, 1),   'expect': (2, 9, 1)},

    {'slice': (None, 8, 1),  'expect': (0, 8, 1)},
    {'slice': (-3, 8, 1),    'expect': (7, 8, 1)},
    {'slice': (11, 8, 1),    'expect': (10, 10, 1)},
    {'slice': (11, 11, 1),   'expect': (10, 10, 1)},
    {'slice': (-11, 8, 1),   'expect': (0, 8, 1)},
    {'slice': (-11, -11, 1), 'expect': (0, 0, 1)},

    {'slice': (8, 2, -1),    'expect': (8, 2, -1)},
    {'slice': (8, None, -1), 'expect': (8, -1, -1)},
    {'slice': (8, 9, -1),    'expect': (8, 8, -1)},
    {'slice': (8, -3, -1),   'expect': (8, 7, -1)},

    {'slice': (None, 8, -1),  'expect': (9, 8, -1)},
    {'slice': (-3, 6, -1),    'expect': (7, 6, -1)},

    {'slice': (10, 10, -1),   'expect': (9, 9, -1)},
    {'slice': (10, 8, -1),    'expect': (9, 8, -1)},
    {'slice': (9, 10, -1),    'expect': (9, 9, -1)},
    {'slice': (9, 9, -1),     'expect': (9, 9, -1)},
    {'slice': (9, 8, -1),     'expect': (9, 8, -1)},
    {'slice': (8, 8, -1),     'expect': (8, 8, -1)},
    {'slice': (-9, -8, -1),   'expect': (1, 1, -1)},
    {'slice': (-9, -9, -1),   'expect': (1, 1, -1)},
    {'slice': (-9, -10, -1),  'expect': (1, 0, -1)},
    {'slice': (-9, -11, -1),  'expect': (1, -1, -1)},
    {'slice': (-9, -12, -1),  'expect': (1, -1, -1)},
    {'slice': (-10, -9, -1),  'expect': (0, 0, -1)},
    {'slice': (-10, -10, -1), 'expect': (0, 0, -1)},
    {'slice': (-10, -11, -1), 'expect': (0, -1, -1)},
    {'slice': (-10, -12, -1), 'expect': (0, -1, -1)},
    {'slice': (-11, 8, -1),   'expect': (-1, -1, -1)},
    {'slice': (-11, -9, -1),  'expect': (-1, -1, -1)},
    {'slice': (-11, -10, -1), 'expect': (-1, -1, -1)},
    {'slice': (-11, -11, -1), 'expect': (-1, -1, -1)},
    {'slice': (-11, -12, -1), 'expect': (-1, -1, -1)},
)
class TestCompleteSlice(unittest.TestCase):

    def test_complete_slice(self):
        assert internal.complete_slice(
            slice(*self.slice), 10) == slice(*self.expect)


class TestCompleteSliceError(unittest.TestCase):

    def test_invalid_step_value(self):
        with self.assertRaises(ValueError):
            internal.complete_slice(slice(1, 1, 0), 1)

    def test_invalid_step_type(self):
        with self.assertRaises(TypeError):
            internal.complete_slice(slice(1, 1, (1, 2)), 1)

    def test_invalid_start_type(self):
        with self.assertRaises(TypeError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with self.assertRaises(TypeError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)

    def test_invalid_stop_type(self):
        with self.assertRaises(TypeError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with self.assertRaises(TypeError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)


@testing.parameterize(
    {'x': 0, 'expect': 0},
    {'x': 1, 'expect': 1},
    {'x': 2, 'expect': 2},
    {'x': 3, 'expect': 4},
    {'x': 2 ** 10,     'expect': 2 ** 10},
    {'x': 2 ** 10 - 1, 'expect': 2 ** 10},
    {'x': 2 ** 10 + 1, 'expect': 2 ** 11},
    {'x': 2 ** 40,     'expect': 2 ** 40},
    {'x': 2 ** 40 - 1, 'expect': 2 ** 40},
    {'x': 2 ** 40 + 1, 'expect': 2 ** 41},
)
class TestClp2(unittest.TestCase):

    def test_clp2(self):
        assert internal.clp2(self.x) == self.expect


@testing.parameterize(*testing.product({
    'value': [0.0, 1.0, -1.0,
              0.25, -0.25,
              11.0, -11.0,
              2 ** -15, -(2 ** -15),  # Denormalized Number
              float('inf'), float('-inf')],
}))
class TestConvertFloat16(unittest.TestCase):

    def test_conversion(self):
        half = internal.to_float16(self.value)
        assert internal.from_float16(half) == self.value


class TestConvertFloat16Nan(unittest.TestCase):

    def test_conversion(self):
        half = internal.to_float16(float('nan'))
        assert math.isnan(internal.from_float16(half))
