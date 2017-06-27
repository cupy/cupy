import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestSort(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_cupy_raises()
    def test_sort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        a.sort()

    @testing.numpy_cupy_raises()
    def test_external_sort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.sort(a)

    def test_sort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            a.sort()

    def test_external_sort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            return cupy.sort(a)

    # Test dtypes

    @testing.for_dtypes(['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q',
                         numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    @testing.for_dtypes(['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q',
                         numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_external_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.sort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(TypeError):
            a.sort()

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_external_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(TypeError):
            return cupy.sort(a)

    # Test contiguous arrays

    @testing.numpy_cupy_allclose()
    def test_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        a.sort()
        return a

    def test_sort_non_contiguous(self):
        a = testing.shaped_random((10,), cupy)[::2]  # Non contiguous view
        with self.assertRaises(ValueError):
            a.sort()

    @testing.numpy_cupy_allclose()
    def test_external_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        return xp.sort(a)

    @testing.numpy_cupy_allclose()
    def test_external_sort_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)[::2]  # Non contiguous view
        return xp.sort(a)


@testing.gpu
class TestLexsort(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_cupy_raises()
    def test_lexsort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.lexsort(a)

    @testing.numpy_cupy_array_equal
    def test_lexsort_one_dim(self, xp):
        a = testing.shaped_random((2,), xp)
        return xp.lexsort(a)

    @testing.numpy_cupy_array_equal
    def test_lexsort_two_dim(self, xp):
        a = xp.array([[9, 4, 0, 4, 0, 2, 1],
                      [1, 5, 1, 4, 3, 4, 4]])  # from numpy.lexsort example
        return xp.lexsort(a)

    def test_lexsort_three_or_more_dim(self):
        a = testing.shaped_random((2, 10, 10), cupy)
        with self.assertRaises(NotImplementedError):
            return cupy.lexsort(a)

    # Test dtypes

    @testing.for_dtypes(['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L',
                         numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_lexsort_dtype(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        return xp.lexsort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_lexsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((2, 10), cupy, dtype)
        with self.assertRaises(TypeError):
            return cupy.lexsort(a)


@testing.gpu
class TestArgsort(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_cupy_raises()
    def test_argsort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return a.argsort()

    @testing.numpy_cupy_raises()
    def test_external_argsort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.argsort(a)

    def test_argsort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            return a.argsort()

    def test_external_argsort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            return cupy.argsort(a)

    # Test dtypes

    @testing.for_dtypes(['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q',
                         numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_argsort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return a.argsort()

    @testing.for_dtypes(['b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q',
                         numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose()
    def test_external_argsort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.argsort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(TypeError):
            return a.argsort()

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_external_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(TypeError):
            return cupy.argsort(a)

    def test_argsort_keep_original_array(self):
        a = testing.shaped_random((10,), cupy)
        b = cupy.array(a)
        a.argsort()
        testing.assert_allclose(a, b)
