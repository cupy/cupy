import operator
import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestArrayBoolOp(unittest.TestCase):

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

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def check_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_op_full(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return op(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_neg_array(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return operator.neg(a)

    def test_pos_array(self):
        self.check_array_op(operator.pos)

    @testing.with_requires('numpy<1.16')
    def test_pos_array_full(self):
        self.check_array_op_full(operator.pos)

    def test_abs_array(self):
        self.check_array_op_full(operator.abs)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def check_zerodim_op(self, op, xp, dtype):
        a = xp.array(-2, dtype)
        return op(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_zerodim_op_full(self, op, xp, dtype):
        a = xp.array(-2, dtype)
        return op(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_neg_zerodim(self, xp, dtype):
        a = xp.array(-2, dtype)
        return operator.neg(a)

    def test_pos_zerodim(self):
        self.check_zerodim_op(operator.pos)

    def test_abs_zerodim(self):
        self.check_zerodim_op_full(operator.abs)

    @testing.with_requires('numpy<1.16')
    def test_abs_zerodim_full(self):
        self.check_zerodim_op_full(operator.abs)


@testing.gpu
class TestArrayIntUnaryOp(unittest.TestCase):

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
