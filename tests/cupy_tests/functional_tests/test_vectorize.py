import unittest

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


class TestVectorize(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
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
