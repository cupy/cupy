import unittest

import numpy
import pytest

import cupy
from cupy import testing


class TestVectorizeOps(unittest.TestCase):

    def _run(self, func, xp, dtypes):
        f = xp.vectorize(func)
        args = [
            testing.shaped_random((20, 30), xp, dtype, seed=seed)
            for seed, dtype in enumerate(dtypes)
        ]
        return f(*args)

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal()
    def test_vectorize_add(self, xp, dtype1, dtype2):
        def my_add(x, y):
            return x + y

        return self._run(my_add, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_sub(self, xp, dtype1, dtype2):
        def my_sub(x, y):
            return x - y

        return self._run(my_sub, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_vectorize_mul(self, xp, dtype1, dtype2):
        def my_mul(x, y):
            return x * y

        return self._run(my_mul, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_vectorize_pow(self, xp, dtype1, dtype2):
        def my_pow(x, y):
            return x ** y

        f = xp.vectorize(my_pow)
        x1 = testing.shaped_random((20, 30), xp, dtype1, seed=0)
        x2 = testing.shaped_random((20, 30), xp, dtype2, seed=1)
        x1[x1 == 0] = 1
        return f(x1, x2)

    def run_div(self, func, xp, dtypes):
        dtype1, dtype2 = dtypes
        f = xp.vectorize(func)
        x1 = testing.shaped_random((20, 30), xp, dtype1, seed=0)
        x2 = testing.shaped_random((20, 30), xp, dtype2, seed=1)
        x2[x2 == 0] = 1
        return f(x1, x2)

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_vectorize_div(self, xp, dtype1, dtype2):
        def my_div(x, y):
            return x / y

        return self.run_div(my_div, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_vectorize_floor_div(self, xp, dtype1, dtype2):
        def my_floor_div(x, y):
            return x // y

        return self.run_div(my_floor_div, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_vectorize_mod(self, xp, dtype1, dtype2):
        def my_mod(x, y):
            return x % y

        return self.run_div(my_mod, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_lshift(self, xp, dtype1, dtype2):
        def my_lshift(x, y):
            return x << y

        return self._run(my_lshift, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_rshift(self, xp, dtype1, dtype2):
        def my_lshift(x, y):
            return x >> y

        return self._run(my_lshift, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_bit_or(self, xp, dtype1, dtype2):
        def my_bit_or(x, y):
            return x | y

        return self._run(my_bit_or, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_bit_and(self, xp, dtype1, dtype2):
        def my_bit_and(x, y):
            return x & y

        return self._run(my_bit_and, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_bit_xor(self, xp, dtype1, dtype2):
        def my_bit_xor(x, y):
            return x ^ y

        return self._run(my_bit_xor, xp, [dtype1, dtype2])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_bit_invert(self, xp, dtype):
        def my_bit_invert(x):
            return ~x

        return self._run(my_bit_invert, xp, [dtype])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_logical_not(self, xp, dtype):
        def my_logical_not(x):
            return not x

        return self._run(my_logical_not, xp, [dtype])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_eq(self, xp, dtype1, dtype2):
        def my_eq(x, y):
            return x == y

        return self._run(my_eq, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_neq(self, xp, dtype1, dtype2):
        def my_neq(x, y):
            return x != y

        return self._run(my_neq, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_lt(self, xp, dtype1, dtype2):
        def my_lt(x, y):
            return x < y

        return self._run(my_lt, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_le(self, xp, dtype1, dtype2):
        def my_le(x, y):
            return x <= y

        return self._run(my_le, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_gt(self, xp, dtype1, dtype2):
        def my_gt(x, y):
            return x > y

        return self._run(my_gt, xp, [dtype1, dtype2])

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_ge(self, xp, dtype1, dtype2):
        def my_ge(x, y):
            return x >= y

        return self._run(my_ge, xp, [dtype1, dtype2])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_usub(self, xp, dtype):
        def my_usub(x):
            return -x

        return self._run(my_usub, xp, [dtype])


class TestVectorizeExprs(unittest.TestCase):

    @testing.for_all_dtypes(name='cond_dtype', no_complex=True)
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vectorize_ifexp(self, xp, dtype, cond_dtype):
        def my_ifexp(c, x, y):
            return x if c else y

        f = xp.vectorize(my_ifexp)
        cond = testing.shaped_random((20, 30), xp, cond_dtype, seed=0)
        x = testing.shaped_random((20, 30), xp, dtype, seed=1)
        y = testing.shaped_random((20, 30), xp, dtype, seed=2)
        return f(cond, x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vectorize_incr(self, xp, dtype):
        def my_incr(x):
            return x + 1

        f = xp.vectorize(my_incr)
        x = testing.shaped_random((20, 30), xp, dtype, seed=0)
        return f(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_vectorize_ufunc_call(self, xp, dtype):
        def my_ufunc_add(x, y):
            return xp.add(x, y)

        f = xp.vectorize(my_ufunc_add)
        x = testing.shaped_random((20, 30), xp, dtype, seed=1)
        y = testing.shaped_random((20, 30), xp, dtype, seed=2)
        return f(x, y)

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(
        rtol={numpy.float16: 1e3, 'default': 1e-7}, accept_error=TypeError)
    def test_vectorize_ufunc_call_dtype(self, xp, dtype1, dtype2):
        def my_ufunc_add(x, y):
            return xp.add(x, y, dtype=dtype2)

        f = xp.vectorize(my_ufunc_add)
        x = testing.shaped_random((20, 30), xp, dtype1, seed=1)
        y = testing.shaped_random((20, 30), xp, dtype1, seed=2)
        return f(x, y)

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'), full=True)
    @testing.numpy_cupy_array_equal(
        accept_error=(TypeError, numpy.ComplexWarning))
    def test_vectorize_typecast(self, xp, dtype1, dtype2):
        typecast = xp.dtype(dtype2).type

        def my_typecast(x):
            return typecast(x)

        f = xp.vectorize(my_typecast)
        x = testing.shaped_random((20, 30), xp, dtype1, seed=1)
        return f(x)


class TestVectorizeInstructions(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vectorize_assign_new(self, xp, dtype):
        def my_assign(x):
            y = x + x
            return x + y

        f = xp.vectorize(my_assign)
        x = testing.shaped_random((20, 30), xp, dtype, seed=1)
        return f(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vectorize_assign_update(self, xp, dtype):
        def my_assign(x):
            x = x + x
            return x + x

        f = xp.vectorize(my_assign)
        x = testing.shaped_random((20, 30), xp, dtype, seed=1)
        return f(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vectorize_augassign(self, xp, dtype):
        def my_augassign(x):
            x += x
            return x + x

        f = xp.vectorize(my_augassign)
        x = testing.shaped_random((20, 30), xp, dtype, seed=1)
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_vectorize_const_assign(self, xp):
        def my_typecast(x):
            typecast = xp.dtype('f').type
            return typecast(x)

        f = xp.vectorize(my_typecast)
        x = testing.shaped_random((20, 30), xp, numpy.int32, seed=1)
        return f(x)

    def test_vectorize_const_typeerror(self):
        def my_invalid_type(x):
            x = numpy.dtype('f').type
            return x

        f = cupy.vectorize(my_invalid_type)
        x = testing.shaped_random((20, 30), cupy, numpy.int32, seed=1)
        with pytest.raises(TypeError):
            f(x)

    def test_vectorize_const_non_toplevel(self):
        def my_invalid_type(x):
            if x == 3:
                typecast = numpy.dtype('f').type  # NOQA
            return x

        f = cupy.vectorize(my_invalid_type)
        x = cupy.array([1, 2, 3, 4, 5])
        with pytest.raises(TypeError):
            f(x)

    @testing.numpy_cupy_array_equal()
    def test_vectorize_nonconst_for_value(self, xp):
        def my_nonconst_result(x):
            result = numpy.int32(0)
            result = x
            return result

        f = xp.vectorize(my_nonconst_result)
        x = testing.shaped_random((20, 30), xp, numpy.int32, seed=1)
        return f(x)


class TestVectorizeStmts(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_if(self, xp):
        def func_if(x):
            if x % 2 == 0:
                y = x
            else:
                y = -x
            return y

        f = xp.vectorize(func_if)
        x = xp.array([1, 2, 3, 4, 5])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_if_no_orlese(self, xp):
        def func_if(x):
            y = 0
            if x % 2 == 0:
                y = x
            return y

        f = xp.vectorize(func_if)
        x = xp.array([1, 2, 3, 4, 5])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_elif(self, xp):
        def func_if(x):
            y = 0
            if x % 2 == 0:
                y = x
            elif x % 3 == 0:
                y = -x
            return y

        f = xp.vectorize(func_if)
        x = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_while(self, xp):
        def func_while(x):
            y = 0
            while x > 0:
                y += x
                x -= 1
            return y

        f = xp.vectorize(func_while)
        x = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        return f(x)

    @testing.for_dtypes('qQ')
    @testing.numpy_cupy_array_equal()
    def test_for(self, xp, dtype):
        def func_for(x):
            y = 0
            for i in range(x):
                y += i
            return y

        f = xp.vectorize(func_for)
        x = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype)
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_for_const_range(self, xp):
        def func_for(x):
            for i in range(3, 10):
                x += i
            return x

        f = xp.vectorize(func_for)
        x = xp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_for_range_step(self, xp):
        def func_for(x, y, z):
            res = 0
            for i in range(x, y, z):
                res += i * i
            return x

        f = xp.vectorize(func_for)
        start = xp.array([0, 1, 2, 3, 4, 5])
        stop = xp.array([-21, -23, -19, 17, 27, 24])
        step = xp.array([-3, -2, -1, 1, 2, 3])
        return f(start, stop, step)

    @testing.numpy_cupy_array_equal()
    def test_for_update_counter(self, xp):
        def func_for(x):
            for i in range(10):
                x += i
                i += 1
            return x

        f = xp.vectorize(func_for)
        x = xp.array([0, 1, 2, 3, 4])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_for_counter_after_loop(self, xp):
        def func_for(x):
            for i in range(10):
                pass
            return x + i

        f = xp.vectorize(func_for)
        x = xp.array([0, 1, 2, 3, 4])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_for_compound_expression_param(self, xp):
        def func_for(x, y):
            res = 0
            for i in range(x * y):
                res += i
            return res

        f = xp.vectorize(func_for)
        x = xp.array([0, 1, 2, 3, 4])
        return f(x, x)

    @testing.numpy_cupy_array_equal()
    def test_for_update_loop_condition(self, xp):
        def func_for(x):
            res = 0
            for i in range(x):
                res += i
                x -= 1
            return res

        f = xp.vectorize(func_for)
        x = xp.array([0, 1, 2, 3, 4])
        return f(x)

    @testing.numpy_cupy_array_equal()
    def test_tuple(self, xp):
        def func_tuple(x, y):
            x, y = y, x
            z = x, y
            a, b = z
            return a * a + b

        f = xp.vectorize(func_tuple)
        x = xp.array([0, 1, 2, 3, 4])
        y = xp.array([5, 6, 7, 8, 9])
        return f(x, y)

    @testing.numpy_cupy_array_equal()
    def test_return_tuple(self, xp):
        def func_tuple(x, y):
            return x + y, x / y

        f = xp.vectorize(func_tuple)
        x = xp.array([0, 1, 2, 3, 4])
        y = xp.array([5, 6, 7, 8, 9])
        return f(x, y)


class _MyClass:

    def __init__(self, x):
        self.x = x


class TestVectorizeConstants(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_vectorize_const_value(self, xp):

        def my_func(x1, x2):
            return x1 - x2 + const

        const = 8
        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, xp.int64, seed=1)
        x2 = testing.shaped_random((20, 30), xp, xp.int64, seed=2)
        return f(x1, x2)

    @testing.numpy_cupy_array_equal()
    def test_vectorize_const_attr(self, xp):

        def my_func(x1, x2):
            return x1 - x2 + const.x

        const = _MyClass(10)
        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, xp.int64, seed=1)
        x2 = testing.shaped_random((20, 30), xp, xp.int64, seed=2)
        return f(x1, x2)


class TestVectorizeBroadcast(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_vectorize_broadcast(self, xp, dtype):
        def my_func(x1, x2):
            return x1 + x2

        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, dtype, seed=1)
        x2 = testing.shaped_random((30,), xp, dtype, seed=2)
        return f(x1, x2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_vectorize_python_scalar_input(self, xp, dtype):
        def my_func(x1, x2):
            return x1 + x2

        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, dtype, seed=1)
        x2 = 1
        return f(x1, x2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_vectorize_numpy_scalar_input(self, xp, dtype):
        def my_func(x1, x2):
            return x1 + x2

        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, dtype, seed=1)
        x2 = dtype(1)
        return f(x1, x2)


class TestVectorize(unittest.TestCase):

    @testing.for_dtypes('qQefdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_vectorize_arithmetic_ops(self, xp, dtype):
        def my_func(x1, x2, x3):
            y = x1 + x2 * x3 ** x1
            x2 = y + x3 * x1
            return x1 + x2 + x3

        f = xp.vectorize(my_func)
        x1 = testing.shaped_random((20, 30), xp, dtype, seed=1)
        x2 = testing.shaped_random((20, 30), xp, dtype, seed=2)
        x3 = testing.shaped_random((20, 30), xp, dtype, seed=3)
        return f(x1, x2, x3)
