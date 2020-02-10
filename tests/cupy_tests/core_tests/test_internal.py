import math

import unittest

from cupy.core import internal
from cupy import testing


class TestProd(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(internal.prod([]), 1)

    def test_one(self):
        self.assertEqual(internal.prod([2]), 2)

    def test_two(self):
        self.assertEqual(internal.prod([2, 3]), 6)


class TestProdSequence(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(internal.prod_sequence(()), 1)

    def test_one(self):
        self.assertEqual(internal.prod_sequence((2,)), 2)

    def test_two(self):
        self.assertEqual(internal.prod_sequence((2, 3)), 6)


class TestGetSize(unittest.TestCase):

    def test_none(self):
        self.assertEqual(internal.get_size(None), ())

    def check_collection(self, a):
        self.assertEqual(internal.get_size(a), tuple(a))

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        self.assertEqual(internal.get_size(1), (1,))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            internal.get_size(1.0)


class TestVectorEqual(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(internal.vector_equal([], []), True)

    def test_not_equal(self):
        self.assertEqual(internal.vector_equal([1, 2, 3], [1, 2, 0]), False)

    def test_equal(self):
        self.assertEqual(internal.vector_equal([-1, 0, 1], [-1, 0, 1]), True)

    def test_different_size(self):
        self.assertEqual(internal.vector_equal([1, 2, 3], [1, 2]), False)


class TestGetContiguousStrides(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(internal.get_contiguous_strides((), 1, True), [])

    def test_one(self):
        self.assertEqual(internal.get_contiguous_strides((1,), 2, True), [2])

    def test_two(self):
        self.assertEqual(internal.get_contiguous_strides((1, 2), 3, True),
                         [6, 3])

    def test_three(self):
        self.assertEqual(internal.get_contiguous_strides((1, 2, 3), 4, True),
                         [24, 12, 4])

    def test_zero_f(self):
        self.assertEqual(internal.get_contiguous_strides((), 1, False), [])

    def test_one_f(self):
        self.assertEqual(internal.get_contiguous_strides((1,), 2, False), [2])

    def test_two_f(self):
        self.assertEqual(internal.get_contiguous_strides((1, 2), 3, False),
                         [3, 3])

    def test_three_f(self):
        self.assertEqual(internal.get_contiguous_strides((1, 2, 3), 4, False),
                         [4, 4, 8])


class TestGetCContiguity(unittest.TestCase):

    def test_zero_in_shape(self):
        self.assertTrue(internal.get_c_contiguity((1, 0, 1), (1, 1, 1), 3))

    def test_all_one_shape(self):
        self.assertTrue(internal.get_c_contiguity((1, 1, 1), (1, 1, 1), 3))

    def test_normal1(self):
        self.assertTrue(internal.get_c_contiguity((3, 4, 3), (24, 6, 2), 2))

    def test_normal2(self):
        self.assertTrue(internal.get_c_contiguity((3, 1, 3), (6, 100, 2), 2))

    def test_normal3(self):
        self.assertTrue(internal.get_c_contiguity((3,), (4, ), 4))

    def test_normal4(self):
        self.assertTrue(internal.get_c_contiguity((), (), 4))

    def test_normal5(self):
        self.assertTrue(internal.get_c_contiguity((3, 1), (4, 8), 4))

    def test_no_contiguous1(self):
        self.assertFalse(internal.get_c_contiguity((3, 4, 3), (30, 6, 2), 2))

    def test_no_contiguous2(self):
        self.assertFalse(internal.get_c_contiguity((3, 1, 3), (24, 6, 2), 2))

    def test_no_contiguous3(self):
        self.assertFalse(internal.get_c_contiguity((3, 1, 3), (6, 6, 4), 2))


class TestInferUnknownDimension(unittest.TestCase):

    def test_known_all(self):
        self.assertEqual(internal.infer_unknown_dimension((1, 2, 3), 6),
                         [1, 2, 3])

    def test_multiple_unknown(self):
        with self.assertRaises(ValueError):
            internal.infer_unknown_dimension((-1, 1, -1), 10)

    def test_infer(self):
        self.assertEqual(internal.infer_unknown_dimension((-1, 2, 3), 12),
                         [2, 2, 3])


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
        self.assertEqual(
            internal.complete_slice(slice(*self.slice), 10),
            slice(*self.expect))


@testing.with_requires('numpy>=1.12')
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
