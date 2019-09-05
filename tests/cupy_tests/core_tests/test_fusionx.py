import unittest
import cupy as cp
from cupy import testing

@testing.gpu
class TestElementwise(unittest.TestCase):
    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype1, dtype2):
        a = xp.array([1, 2], dtype=dtype1)
        b = xp.array([3, 4], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_add_2d(self, xp, dtype1, dtype2):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype1)
        b = xp.array([[5, 6], [7, 8]], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast1(self, xp, dtype1, dtype2):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype1)
        b = xp.array([5, 6], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast2(self, xp, dtype1, dtype2):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype1)
        b = xp.array([5, 6], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            a = x + y
            return y

        return f(a, b)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2', 'dtype3'])
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast3(self, xp, dtype1, dtype2, dtype3):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype1)
        b = xp.array([[5, 6], [6, 7], [7, 8]], dtype=dtype2)
        c = xp.array([9, 10], dtype=dtype3)

        @cp.fusex()
        def f(x, y, z):
            a = x + z
            b = y + z
            return z

        return f(a, b, c)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_broadcast4(self, xp, dtype1, dtype2):
        a = xp.array([1], dtype=dtype1)
        b = xp.array([2, 3], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_broadcast5(self, xp, dtype1, dtype2):
        a = xp.array([1], dtype=dtype1)
        b = xp.array([[2, 3], [4, 5]], dtype=dtype2)

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

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_premap_postmap(self, xp, dtype1, dtype2):
        a = xp.array([[0, 1], [2, 3]], dtype=dtype1)
        b = xp.array([1], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            a = x + y
            b = xp.sum(a, axis=0)
            return b + y

        return f(a, b)

@testing.gpu
class TestMultiReduction(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum(self, xp, dtype):
        a = xp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)

        @cp.fusex()
        def f(x):
            a = xp.sum(x, axis=0)
            b = xp.sum(a, axis=0)
            return b

        return f(a)

    @testing.for_int_dtypes_combination(names=['dtype1', 'dtype2', 'dtype3'])
    @testing.numpy_cupy_array_equal()
    def test_premap_postmap(self, xp, dtype1, dtype2, dtype3):
        a = xp.array([1, 2], dtype=dtype1)
        b = xp.array([[1, 2], [3, 4]], dtype=dtype2)
        c = xp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype3)

        @cp.fusex()
        def f(x, y, z):
            a = x + y
            b = xp.sum(z, axis=0) + a
            c = b * x
            d = xp.sum(c, axis=0)
            e = a + d
            return e

        return f(a, b, c)

@testing.gpu
class TestReal(unittest.TestCase):
    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add(self, xp, dtype1, dtype2):
        a = xp.array([0, 1], dtype=dtype1)
        b = xp.array([2, 3], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add_2d(self, xp, dtype1, dtype2):
        a = xp.array([[1.2, 2.3], [3.4, 4.5]], dtype=dtype1)
        b = xp.array([5.6, 6.7], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_reduce(self, xp, dtype):
        a = xp.array([[1.2], [2.3], [3.4]], dtype=dtype)

        @cp.fusex()
        def f(x):
            return xp.sum(x, axis=0)

        return f(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_sqrt(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)

        @cp.fusex()
        def f(x):
            return xp.sqrt(x)

        return f(a)

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_cast_add(self, xp, dtype1, dtype2):
        a = xp.array([[0, 1], [2, 3]], dtype=dtype1)
        b = xp.array([1, 2], dtype=dtype2)

        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_cast_sqrt_add(self, xp, dtype):
        a = xp.array([[1], [2], [3], [4]], dtype=dtype)

        @cp.fusex()
        def f(x):
            a = xp.sqrt(x)
            b = a + x
            return xp.sum(b, axis=0)

        return f(a)

class TestDuplicateArgs(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + x

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add2(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + x + x + x + x

        return f(a)

class TestInplace(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            x += x
            return x

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add2(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        b = xp.array([3, 4], dtype=dtype)
        @cp.fusex()
        def f(x, y):
            x += y
            return x

        return f(a, b)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add_broadcast(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        b = xp.array([5, 6], dtype=dtype)
        @cp.fusex()
        def f(x, y):
            x += y
            return x

        return f(a, b)

class TestReductionAxis(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_reduce(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4], [4, 5]], dtype=dtype)
        @cp.fusex()
        def f(x):
            return xp.sum(x, axis=1)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_reduce_3d(self, xp, dtype):
        a = xp.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=dtype)
        @cp.fusex()
        def f(x):
            return xp.sum(x, axis=2)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_reduce_3d2(self, xp, dtype):
        a = xp.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=dtype)

        @cp.fusex()
        def f(x):
            y = xp.sum(x, axis=2)
            return xp.sum(y, axis=1)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_reduce_misc(self, xp, dtype):
        a = xp.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=dtype)

        @cp.fusex()
        def f(x):
            a = xp.sqrt(x)
            b = xp.sum(a, axis=1)
            c = xp.sqrt(b)
            return xp.sum(c, axis=0)

        return f(a)

class TestFusionVarScalar(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + 2

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add2(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + 2.0

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add3(self, xp, dtype):
        a = xp.array([1, 2], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + 2

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add2d(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        @cp.fusex()
        def f(x):
            return x + 2

        return f(a)

    @testing.for_dtypes(['i', 'l', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add_inplace(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        @cp.fusex()
        def f(x):
            x += 2
            return x

        return f(a)

    @testing.for_dtypes(['i', 'l', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_misc(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        b = xp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        @cp.fusex()
        def f(x, y):
            x += 4
            y += 7
            y += x
            a = xp.sum(y, axis=1)
            b = a * 2
            return xp.sqrt(b)

        return f(a, b)

class TestReductionToZeroDim(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        @cp.fusex()
        def f(x):
            return xp.sum(x)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum2d(self, xp, dtype):
        a = xp.array([[1, 2], [3, 4]], dtype=dtype)
        @cp.fusex()
        def f(x):
            return xp.sum(x)

        return f(a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sum3d(self, xp, dtype):
        a = xp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)
        @cp.fusex()
        def f(x):
            return xp.sum(x)

        return f(a)

    @testing.for_dtypes(['l', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        @cp.fusex()
        def f(x):
            a = xp.sum(x)
            x += a
            return x

        return f(a)

    @testing.for_dtypes(['l', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add2(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        @cp.fusex()
        def f(x):
            a = xp.sum(x)
            x += a
            b = xp.sum(x)
            x += b
            return x

        return f(a)

    @testing.for_dtypes(['l', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add3(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        @cp.fusex()
        def f(x):
            a = xp.sum(x)
            x += a
            b = xp.sum(x)
            return a + b

        return f(a)

    @testing.for_dtypes(['l', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add4(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        @cp.fusex()
        def f(x):
            x += 10
            a = xp.sum(x)
            x += a
            b = xp.sum(x)
            a += b
            a += 10
            return a

        return f(a)

class TestScalarInput(unittest.TestCase):
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_add(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        b = 4
        @cp.fusex()
        def f(x, y):
            return x + y

        return f(a, b)

    @testing.for_dtypes(['l', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_add2(self, xp, dtype):
        a = -1
        b = xp.array([[1, 2], [3, 4]], dtype=dtype)
        c = 10
        @cp.fusex()
        def f(x, y, z):
            y += x
            a = xp.sum(y, axis=1)
            y *= z
            b = xp.sum(y, axis=0)
            return a * b

        return f(a, b, c)
