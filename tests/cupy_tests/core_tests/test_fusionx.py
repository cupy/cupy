import unittest
import cupy as cp
from cupy import testing

@testing.gpu
class TestElementwise(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        b = xp.array([3, 4], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add_2d(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([[5, 6], [7, 8]], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast1(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([5, 6], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast2(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([5, 6], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            a = x + y
            return y

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast3(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([[5, 6], [6, 7], [7, 8]], dtype=dtype)
        c = xp.array([9, 10], dtype=dtype)

        @cp.fusex()
        def f(x, y, z):
            a = x + z
            b = y + z
            return z

        return f(a, b, c)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast4(self, xp, dtype):
        a = xp.array([1], dtype=dtype)
        b = xp.array([2, 3], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast5(self, xp, dtype):
        a = xp.array([1], dtype=dtype)
        b = xp.array([[2, 3], [4, 5]], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

@testing.gpu
class TestSingleReduction(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)

        @cp.fusex()
        def f(x):
            return xp.sum(x, axis=0)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum_2d(self, xp, dtype):
        a = xp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)

        @cp.fusex()
        def f(x):
            return xp.sum(x, axis=0)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_premap_postmap(self, xp, dtype):
        a = xp.array([[0, 1], [2, 3]], dtype=dtype)
        b = xp.array([1], dtype=dtype)

        @cp.fusex()
        def f(x, y):
            a = x + y
            b = xp.sum(a, axis=0)
            return b + y

        return f(a, b)
