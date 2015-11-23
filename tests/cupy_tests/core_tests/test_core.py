import unittest

import numpy

import cupy
from cupy.core import core


class TestGetSize(unittest.TestCase):

    def test_none(self):
        self.assertEqual(core.get_size(None), ())

    def test_list(self):
        self.assertEqual(core.get_size([1, 2]), (1, 2))

    def test_tuple(self):
        self.assertEqual(core.get_size((1, 2)), (1, 2))

    def test_int(self):
        self.assertEqual(core.get_size(1), (1,))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            core.get_size(1.0)


class TestInternalProd(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(core.internal_prod([]), 1)

    def test_one(self):
        self.assertEqual(core.internal_prod([2]), 2)

    def test_two(self):
        self.assertEqual(core.internal_prod([2, 3]), 6)


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


class TestGetContiguousStrides(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(core._get_contiguous_strides((), 1), [])

    def test_one(self):
        self.assertEqual(core._get_contiguous_strides((1,), 2), [2])

    def test_two(self):
        self.assertEqual(core._get_contiguous_strides((1, 2), 3), [6, 3])

    def test_three(self):
        self.assertEqual(core._get_contiguous_strides((1, 2, 3), 4),
                         [24, 12, 4])


class TestGetCContiguity(unittest.TestCase):

    def test_zero_in_shape(self):
        self.assertTrue(core._get_c_contiguity((1, 0, 1), (1, 1, 1), 3))

    def test_normal(self):
        # TODO(unno): write test for normal case
        pass


class TestInferUnknownDimension(unittest.TestCase):

    def test_known_all(self):
        self.assertEqual(core._infer_unknown_dimension((1, 2, 3), 6),
                         [1, 2, 3])

    def test_multiple_unknown(self):
        with self.assertRaises(ValueError):
            core._infer_unknown_dimension((-1, 1, -1), 10)

    def test_infer(self):
        self.assertEqual(core._infer_unknown_dimension((-1, 2, 3), 12),
                         [2, 2, 3])


class TestArray(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2,3), dtype=object)
        with self.assertRaises(ValueError):
            core.array(arr)
