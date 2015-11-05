import unittest

from cupy import internal
from cupy import testing


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
    {'slice': (11, 8, -1),    'expect': (10, 8, -1)},
    {'slice': (11, 11, -1),   'expect': (10, 10, -1)},
    {'slice': (-11, 8, -1),   'expect': (0, 0, -1)},
    {'slice': (-11, -11, -1), 'expect': (0, 0, -1)},
)
class TestCompleteSlice(unittest.TestCase):

    def test_complete_slice(self):
        self.assertEqual(
            internal.complete_slice(slice(*self.slice), 10),
            slice(*self.expect))


class TestCompleteSliceError(unittest.TestCase):

    def test_invalid_step_value(self):
        with self.assertRaises(ValueError):
            internal.complete_slice(slice(1, 1, 0), 1)

    def test_invalid_step_type(self):
        with self.assertRaises(IndexError):
            internal.complete_slice(slice(1, 1, (1, 2)), 1)

    def test_invalid_start_type(self):
        with self.assertRaises(IndexError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with self.assertRaises(IndexError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)

    def test_invalid_stop_type(self):
        with self.assertRaises(IndexError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with self.assertRaises(IndexError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)
