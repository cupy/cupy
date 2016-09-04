import operator
import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestArrayBoolOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_bool_empty(self, dtype):
        self.assertFalse(bool(cupy.array((), dtype=dtype)))

    def test_bool_scalar_bool(self):
        self.assertTrue(bool(cupy.array(True, dtype=numpy.bool)))
        self.assertFalse(bool(cupy.array(False, dtype=numpy.bool)))

    @testing.for_all_dtypes()
    def test_bool_scalar(self, dtype):
        self.assertTrue(bool(cupy.array(1, dtype=dtype)))
        self.assertFalse(bool(cupy.array(0, dtype=dtype)))

    def test_bool_one_element_bool(self):
        self.assertTrue(bool(cupy.array([True], dtype=numpy.bool)))
        self.assertFalse(bool(cupy.array([False], dtype=numpy.bool)))

    @testing.for_all_dtypes()
    def test_bool_one_element(self, dtype):
        self.assertTrue(bool(cupy.array([1], dtype=dtype)))
        self.assertFalse(bool(cupy.array([0], dtype=dtype)))

    @testing.for_all_dtypes()
    def test_bool_two_elements(self, dtype):
        with self.assertRaises(ValueError):
            bool(cupy.array([1, 2], dtype=dtype))


@testing.gpu
class TestArrayUnaryOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    def test_neg_array(self):
        self.check_array_op(operator.neg)

    def test_pos_array(self):
        self.check_array_op(operator.pos)

    def test_abs_array(self):
        self.check_array_op(operator.abs)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_zerodim_op(self, op, xp, dtype):
        a = xp.array(-2, dtype)
        return op(a)

    def test_neg_zerodim(self):
        self.check_zerodim_op(operator.neg)

    def test_pos_zerodim(self):
        self.check_zerodim_op(operator.pos)

    def test_abs_zerodim(self):
        self.check_zerodim_op(operator.abs)


@testing.gpu
class TestArrayIntUnaryOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    def test_invert_array(self):
        self.check_array_op(operator.invert)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_zerodim_op(self, op, xp, dtype):
        a = xp.array(-2, dtype)
        return op(a)

    def test_invert_zerodim(self):
        self.check_zerodim_op(operator.invert)
