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

    @testing.numpy_cupy_array_equal()
    def test_sort_two_or_more_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort()
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_two_or_more_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_external_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.sort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            a.sort()

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_external_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return cupy.sort(a)

    # Test contiguous arrays

    @testing.numpy_cupy_allclose()
    def test_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        a.sort()
        return a

    def test_sort_non_contiguous(self):
        a = testing.shaped_random((10,), cupy)[::2]  # Non contiguous view
        with self.assertRaises(NotImplementedError):
            a.sort()

    @testing.numpy_cupy_allclose()
    def test_external_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        return xp.sort(a)

    @testing.numpy_cupy_allclose()
    def test_external_sort_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)[::2]  # Non contiguous view
        return xp.sort(a)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=0)
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-2)
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=-2)

    @testing.numpy_cupy_array_equal()
    def test_external_sort_none_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=None)

    @testing.numpy_cupy_raises()
    def test_sort_invalid_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=3)

    @testing.numpy_cupy_raises()
    def test_external_sort_invalid_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=3)

    @testing.numpy_cupy_raises()
    def test_sort_invalid_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-4)

    @testing.numpy_cupy_raises()
    def test_external_sort_invalid_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=-4)


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

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
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
        with self.assertRaises(NotImplementedError):
            return a.argsort()

    def test_external_argsort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(NotImplementedError):
            return cupy.argsort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_argsort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return a.argsort()

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_external_argsort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.argsort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return a.argsort()

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_external_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return cupy.argsort(a)

    def test_argsort_keep_original_array(self):
        a = testing.shaped_random((10,), cupy)
        b = cupy.array(a)
        a.argsort()
        testing.assert_allclose(a, b)


@testing.gpu
class TestMsort(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_cupy_raises()
    def test_msort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.msort(a)

    def test_msort_two_or_more_dim(self):
        a = testing.shaped_random((2, 3), cupy)
        with self.assertRaises(ValueError):
            return cupy.msort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_msort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.msort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_msort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return cupy.msort(a)


@testing.parameterize(*testing.product({
    'external': [False, True],
}))
@testing.gpu
class TestPartition(unittest.TestCase):

    _multiprocess_can_split_ = True

    def partition(self, a, kth, axis=-1):
        if self.external:
            xp = cupy.get_array_module(a)
            return xp.partition(a, kth, axis=axis)
        else:
            a.partition(kth, axis=axis)
            return a

    # Test base cases

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_raises()
    def test_partition_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        kth = 2
        return self.partition(a, kth)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_equal()
    def test_partition_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        self.assertTrue(xp.all(x[0:kth] <= x[kth]))
        self.assertTrue(xp.all(x[kth] <= x[kth + 1:]))
        return x[kth]

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_equal()
    def test_partition_multi_dim(self, xp, dtype):
        a = testing.shaped_random((10, 10, 10), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        self.assertTrue(xp.all(x[:, :, 0:kth] <= x[:, :, kth]))
        self.assertTrue(xp.all(x[:, :, kth] <= x[:, :, kth + 1:]))
        return x[:, :, kth]

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_partition_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        kth = 2
        with self.assertRaises(NotImplementedError):
            return self.partition(a, kth)

    # Test non-contiguous array

    def test_partition_non_contiguous(self):
        a = testing.shaped_random((10,), cupy)[::2]
        kth = 2
        with self.assertRaises(NotImplementedError):
            return self.partition(a, kth)

    # Test kth

    @testing.numpy_cupy_equal()
    def test_partition_sequence_kth(self, xp):
        a = testing.shaped_random((10,), xp)
        kth = (2, 4)
        x = self.partition(a, kth)
        return x[kth[0]], x[kth[1]]

    @testing.numpy_cupy_equal()
    def test_partition_negative_kth(self):
        a = testing.shaped_random((10,), cupy)
        kth = -3
        x = self.partition(a, kth)
        return x[kth]

    @testing.numpy_cupy_raises()
    def test_partition_invalid_kth(self, xp):
        a = testing.shaped_random((10,), xp)
        kth = 10
        return self.partition(a, kth)

    @testing.numpy_cupy_raises()
    def test_partition_invalid_negative_kth(self, xp):
        a = testing.shaped_random((10,), xp)
        kth = -11
        return self.partition(a, kth)

    # Test axis

    @testing.numpy_cupy_equal()
    def test_partition_axis(self, xp):
        a = testing.shaped_random((10, 10, 10), xp)
        kth = 2
        axis = 0
        x = self.partition(a, kth, axis=axis)
        return x[kth, :, :]

    @testing.numpy_cupy_equal()
    def test_partition_negative_axis(self, xp):
        a = testing.shaped_random((10, 10, 10), xp)
        kth = 2
        axis = -1
        x = self.partition(a, kth, axis=axis)
        return x[:, :, kth]

    @testing.numpy_cupy_equal()
    def test_partition_none_axis(self, xp):
        a = testing.shaped_random((2, 2), xp)
        kth = 2
        axis = None
        x = self.partition(a, kth, axis=axis)
        return x[kth]

    @testing.numpy_cupy_raises()
    def test_partition_invalid_axis(self, xp):
        a = testing.shaped_random((2, 2, 2), xp)
        kth = 2
        axis = 3
        return self.partition(a, kth, axis=axis)

    @testing.numpy_cupy_raises()
    def test_partition_invalid_negative_axis(self, xp):
        a = testing.shaped_random((2, 2, 2), xp)
        kth = 2
        axis = -4
        return self.partition(a, kth, axis=axis)
