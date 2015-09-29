import operator
import unittest

import numpy
import six

import cupy
from cupy import testing


@testing.gpu
class TestArrayElementwiseOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_scalar_op(self, op, xp, dtype, swap=False):
        a = testing.shaped_arange((2, 3), xp, dtype)
        if swap:
            return op(dtype(2), a)
        else:
            return op(a, dtype(2))

    def test_add_scalar(self):
        self.check_array_scalar_op(operator.add)

    def test_radd_scalar(self):
        self.check_array_scalar_op(operator.add, swap=True)

    def test_iadd_scalar(self):
        self.check_array_scalar_op(operator.iadd)

    def test_sub_scalar(self):
        self.check_array_scalar_op(operator.sub)

    def test_rsub_scalar(self):
        self.check_array_scalar_op(operator.sub, swap=True)

    def test_isub_scalar(self):
        self.check_array_scalar_op(operator.isub)

    def test_mul_scalar(self):
        self.check_array_scalar_op(operator.mul)

    def test_rmul_scalar(self):
        self.check_array_scalar_op(operator.mul, swap=True)

    def test_imul_scalar(self):
        self.check_array_scalar_op(operator.imul)

    def test_truediv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.truediv)

    def test_rtruediv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.truediv, swap=True)

    def test_itruediv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.itruediv)

    def test_div_scalar(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.div)

    def test_rdiv_scalar(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.div, swap=True)

    def test_idiv_scalar(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.idiv)

    def test_floordiv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.floordiv)

    def test_rfloordiv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.floordiv, swap=True)

    def test_ifloordiv_scalar(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.ifloordiv)

    def test_pow_scalar(self):
        self.check_array_scalar_op(operator.pow)

    def test_rpow_scalar(self):
        self.check_array_scalar_op(operator.pow, swap=True)

    def test_ipow_scalar(self):
        self.check_array_scalar_op(operator.ipow)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return op(a, b)

    def test_add_array(self):
        self.check_array_scalar_op(operator.add)

    def test_iadd_array(self):
        self.check_array_scalar_op(operator.iadd)

    def test_sub_array(self):
        self.check_array_scalar_op(operator.sub)

    def test_isub_array(self):
        self.check_array_scalar_op(operator.isub)

    def test_mul_array(self):
        self.check_array_scalar_op(operator.mul)

    def test_imul_array(self):
        self.check_array_scalar_op(operator.imul)

    def test_truediv_array(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.truediv)

    def test_itruediv_array(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.itruediv)

    def test_div_array(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.div)

    def test_idiv_array(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.idiv)

    def test_floordiv_array(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.floordiv)

    def test_ifloordiv_array(self):
        numpy.seterr(divide='ignore')
        self.check_array_scalar_op(operator.ifloordiv)

    def test_pow_array(self):
        self.check_array_scalar_op(operator.pow)

    def test_ipow_array(self):
        self.check_array_scalar_op(operator.ipow)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_broadcasted_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), dtype=dtype)
        b = testing.shaped_arange((2, 1), dtype=dtype)
        return op(a, b)

    def test_broadcasted_add(self):
        self.check_array_broadcasted_op(operator.add)

    def test_broadcasted_iadd(self):
        self.check_array_broadcasted_op(operator.iadd)

    def test_broadcasted_sub(self):
        self.check_array_broadcasted_op(operator.sub)

    def test_broadcasted_isub(self):
        self.check_array_broadcasted_op(operator.isub)

    def test_broadcasted_mul(self):
        self.check_array_broadcasted_op(operator.mul)

    def test_broadcasted_imul(self):
        self.check_array_broadcasted_op(operator.imul)

    def test_broadcasted_truediv(self):
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.truediv)

    def test_broadcasted_itruediv(self):
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.itruediv)

    def test_broadcasted_div(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.div)

    def test_broadcasted_idiv(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.idiv)

    def test_broadcasted_floordiv(self):
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.floordiv)

    def test_broadcasted_ifloordiv(self):
        numpy.seterr(divide='ignore')
        self.check_array_broadcasted_op(operator.ifloordiv)

    def test_broadcasted_pow(self):
        self.check_array_broadcasted_op(operator.pow)

    def test_broadcasted_ipow(self):
        self.check_array_broadcasted_op(operator.ipow)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_doubly_broadcasted_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 1, 3), xp, dtype)
        b = testing.shaped_arange((3, 1), xp, dtype)
        return op(a, b)

    def test_doubly_broadcasted_add(self):
        self.check_array_doubly_broadcasted_op(operator.add)

    def test_doubly_broadcasted_sub(self):
        self.check_array_doubly_broadcasted_op(operator.sub)

    def test_doubly_broadcasted_mul(self):
        self.check_array_doubly_broadcasted_op(operator.mul)

    def test_doubly_broadcasted_truediv(self):
        numpy.seterr(divide='ignore', invalid='ignore')
        self.check_array_doubly_broadcasted_op(operator.truediv)

    def test_doubly_broadcasted_floordiv(self):
        numpy.seterr(divide='ignore')
        self.check_array_doubly_broadcasted_op(operator.floordiv)

    def test_doubly_broadcasted_div(self):
        if six.PY3:
            return
        numpy.seterr(divide='ignore')
        self.check_array_doubly_broadcasted_op(operator.div)

    def test_doubly_broadcasted_pow(self):
        self.check_array_doubly_broadcasted_op(operator.pow)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_reversed_op(self, op, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return op(a, a[::-1])

    def test_array_reversed_add(self):
        self.check_array_reversed_op(operator.add)

    def test_array_reversed_sub(self):
        self.check_array_reversed_op(operator.sub)

    def test_array_reversed_mul(self):
        self.check_array_reversed_op(operator.mul)

    @testing.for_all_dtypes(no_bool=True)
    def check_typecast(self, val, dtype):
        operators = [operator.add, operator.sub, operator.mul, operator.div]
        for op in operators:
            err = numpy.geterr()
            numpy.seterr(divide='ignore', invalid='ignore')
            a = op(val, (testing.shaped_arange((5,), numpy, dtype) - 2))
            numpy.seterr(**err)
            b = op(val, (testing.shaped_arange((5,), cupy, dtype) - 2))
            self.assertEqual(a.dtype, b.dtype)

    def test_typecast_bool1(self):
        self.check_typecast(True)

    def test_typecast_bool2(self):
        self.check_typecast(False)

    def test_typecast_int1(self):
        self.check_typecast(0)

    def test_typecast_int2(self):
        self.check_typecast(-127)

    def test_typecast_int3(self):
        self.check_typecast(255)

    def test_typecast_int4(self):
        self.check_typecast(-32768)

    def test_typecast_int5(self):
        self.check_typecast(65535)

    def test_typecast_int6(self):
        self.check_typecast(-2147483648)

    def test_typecast_int7(self):
        self.check_typecast(4294967295)

    def test_typecast_float1(self):
        self.check_typecast(0.0)

    def test_typecast_float2(self):
        self.check_typecast(100000.0)


@testing.gpu
class TestArrayIntElementwiseOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_scalar_op(self, op, xp, dtype, swap=False):
        a = testing.shaped_arange((2, 3), xp, dtype)
        if swap:
            return op(dtype(2), a)
        else:
            return op(a, dtype(2))

    def test_lshift_scalar(self):
        self.check_array_scalar_op(operator.lshift)

    def test_rlshift_scalar(self):
        self.check_array_scalar_op(operator.lshift, swap=True)

    def test_rshift_scalar(self):
        self.check_array_scalar_op(operator.rshift)

    def test_rrshift_scalar(self):
        self.check_array_scalar_op(operator.rshift, swap=True)

    def test_and_scalar(self):
        self.check_array_scalar_op(operator.and_)

    def test_rand_scalar(self):
        self.check_array_scalar_op(operator.and_, swap=True)

    def test_or_scalar(self):
        self.check_array_scalar_op(operator.or_)

    def test_ror_scalar(self):
        self.check_array_scalar_op(operator.or_, swap=True)

    def test_xor_scalar(self):
        self.check_array_scalar_op(operator.xor)

    def test_rxor_scalar(self):
        self.check_array_scalar_op(operator.xor, swap=True)

    def test_mod_scalar(self):
        self.check_array_scalar_op(operator.mod)

    def test_rmod_scalar(self):
        self.check_array_scalar_op(operator.mod, swap=True)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_array_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return op(a, b)

    def test_lshift_array(self):
        self.check_array_scalar_op(operator.lshift)

    def test_ilshift_array(self):
        self.check_array_scalar_op(operator.ilshift)

    def test_rshift_array(self):
        self.check_array_scalar_op(operator.rshift)

    def test_irshift_array(self):
        self.check_array_scalar_op(operator.irshift)

    def test_and_array(self):
        self.check_array_scalar_op(operator.and_)

    def test_iand_array(self):
        self.check_array_scalar_op(operator.iand)

    def test_or_array(self):
        self.check_array_scalar_op(operator.or_)

    def test_ior_array(self):
        self.check_array_scalar_op(operator.ior)

    def test_xor_array(self):
        self.check_array_scalar_op(operator.xor)

    def test_ixor_array(self):
        self.check_array_scalar_op(operator.ixor)

    def test_mod_array(self):
        self.check_array_scalar_op(operator.mod)

    def test_imod_array(self):
        self.check_array_scalar_op(operator.imod)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_broadcasted_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 3), dtype=dtype)
        b = testing.shaped_arange((2, 1), dtype=dtype)
        return op(a, b)

    def test_broadcasted_lshift(self):
        self.check_array_broadcasted_op(operator.lshift)

    def test_broadcasted_ilshift(self):
        self.check_array_broadcasted_op(operator.ilshift)

    def test_broadcasted_rshift(self):
        self.check_array_broadcasted_op(operator.rshift)

    def test_broadcasted_irshift(self):
        self.check_array_broadcasted_op(operator.irshift)

    def test_broadcasted_and(self):
        self.check_array_broadcasted_op(operator.and_)

    def test_broadcasted_iand(self):
        self.check_array_broadcasted_op(operator.iand)

    def test_broadcasted_or(self):
        self.check_array_broadcasted_op(operator.or_)

    def test_broadcasted_ior(self):
        self.check_array_broadcasted_op(operator.ior)

    def test_broadcasted_xor(self):
        self.check_array_broadcasted_op(operator.xor)

    def test_broadcasted_ixor(self):
        self.check_array_broadcasted_op(operator.ixor)

    def test_broadcasted_mod(self):
        self.check_array_broadcasted_op(operator.mod)

    def test_broadcasted_imod(self):
        self.check_array_broadcasted_op(operator.imod)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose()
    def check_array_doubly_broadcasted_op(self, op, xp, dtype):
        a = testing.shaped_arange((2, 1, 3), xp, dtype)
        b = testing.shaped_arange((3, 1), xp, dtype)
        return op(a, b)

    def test_doubly_broadcasted_lshift(self):
        self.check_array_doubly_broadcasted_op(operator.lshift)

    def test_doubly_broadcasted_rshift(self):
        self.check_array_doubly_broadcasted_op(operator.rshift)

    def test_doubly_broadcasted_and(self):
        self.check_array_doubly_broadcasted_op(operator.and_)

    def test_doubly_broadcasted_or(self):
        self.check_array_doubly_broadcasted_op(operator.or_)

    def test_doubly_broadcasted_xor(self):
        self.check_array_doubly_broadcasted_op(operator.xor)

    def test_doubly_broadcasted_mod(self):
        self.check_array_doubly_broadcasted_op(operator.mod)
