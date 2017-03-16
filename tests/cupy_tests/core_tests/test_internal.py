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


class TestProdSsizeT(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(internal.prod([]), 1)

    def test_one(self):
        self.assertEqual(internal.prod([2]), 2)

    def test_two(self):
        self.assertEqual(internal.prod([2, 3]), 6)


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

    def test_normal(self):
        # TODO(unno): write test for normal case
        pass


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
