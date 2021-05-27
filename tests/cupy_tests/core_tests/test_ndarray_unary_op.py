import operator
import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


@testing.gpu
class TestArrayBoolOp(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_bool_empty(self, dtype):
        with testing.assert_warns(DeprecationWarning):
            assert not bool(cupy.array((), dtype=dtype))

    def test_bool_scalar_bool(self):
        assert bool(cupy.array(True, dtype=numpy.bool_))
        assert not bool(cupy.array(False, dtype=numpy.bool_))

    @testing.for_all_dtypes()
    def test_bool_scalar(self, dtype):
        assert bool(cupy.array(1, dtype=dtype))
        assert not bool(cupy.array(0, dtype=dtype))

    def test_bool_one_element_bool(self):
        assert bool(cupy.array([True], dtype=numpy.bool_))
        assert not bool(cupy.array([False], dtype=numpy.bool_))

    @testing.for_all_dtypes()
    @pytest.mark.xfail(runtime.is_hip, reason='ROCm/HIP may have a bug')
    def test_bool_one_element(self, dtype):
        assert bool(cupy.array([1], dtype=dtype))
        assert not bool(cupy.array([0], dtype=dtype))

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


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
    'shape': [(3, 2), (), (3, 0, 2)]
}))
@testing.gpu
class TestBoolNeg(unittest.TestCase):

    def test_bool_neg(self):
        xp = self.xp
        if xp is numpy and not testing.numpy_satisfies('>=1.13.0'):
            raise unittest.SkipTest('NumPy<1.13.0')
        shape = self.shape
        x = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        with pytest.raises(TypeError):
            -x
