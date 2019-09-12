import unittest

from cupy import testing


@testing.gpu
class TestAppend(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test(self, xp):
        a = testing.shaped_random((3, 4, 5), xp, xp.float32)
        b = testing.shaped_random((6, 7), xp, xp.float32)
        return xp.append(a, b)

    @testing.numpy_cupy_array_equal()
    def test_scalar_lhs(self, xp):
        return xp.append(10, xp.arange(20))

    @testing.numpy_cupy_array_equal()
    def test_scalar_rhs(self, xp):
        return xp.append(xp.arange(20), 10)

    @testing.numpy_cupy_array_equal()
    def test_scalar_both(self, xp):
        return xp.append(10, 10)

    @testing.numpy_cupy_array_equal()
    def test_axis(self, xp):
        a = testing.shaped_random((3, 4, 5), xp, xp.float32)
        b = testing.shaped_random((3, 10, 5), xp, xp.float32)
        return xp.append(a, b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.append(xp.array(0), xp.arange(10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.append(xp.array([]), xp.arange(10))


@testing.gpu
class TestResize(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test(self, xp):
        return xp.resize(xp.arange(10), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_remainder(self, xp):
        return xp.resize(xp.arange(8), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_shape_int(self, xp):
        return xp.resize(xp.arange(10), 15)

    @testing.numpy_cupy_array_equal()
    def test_scalar(self, xp):
        return xp.resize(2, (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_scalar_shape_int(self, xp):
        return xp.resize(2, 10)

    @testing.numpy_cupy_array_equal()
    def test_typed_scalar(self, xp):
        return xp.resize(xp.float32(10.0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.resize(xp.array(0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.resize(xp.array([]), (10, 10))


@testing.gpu
class TestUnique(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_index(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_index=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_inverse(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_inverse=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_counts(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_counts=True)[1]
