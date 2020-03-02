import unittest
import cupy

from cupy import testing


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

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_append1(self, xp, dtype):
        a = testing.shaped_arange((5, 3, 4), xp, dtype)
        b = testing.shaped_arange((1, 3, 4), xp, dtype)
        return xp.append(a, b, axis=0)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_append2(self, xp, dtype):
        a = testing.shaped_arange((5, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 3, 4), xp, dtype)
        return xp.append(a, b, axis=0)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_append_flatten(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((1, 3), xp, dtype)
        return xp.append(a, b)

    def test_append_wrong_ndim1(self):
        a = cupy.empty((1, 2, 3))
        b = cupy.empty((1, 3))
        with self.assertRaises(ValueError):
            cupy.append(a, b, axis=0)

    def test_append_wrong_ndim2(self):
        a = cupy.empty((2, 3))
        b = cupy.empty((2,))
        with self.assertRaises(ValueError):
            cupy.append(a, b, axis=0)

    def test_append_wrong_shape(self):
        a = cupy.empty((2, 3, 4))
        b = cupy.empty((3, 3, 3))
        with self.assertRaises(ValueError):
            cupy.append(a, b, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_append_many_multi_dptye(self, xp):
        a = testing.shaped_arange((2, 1), xp, 'i')
        b = testing.shaped_arange((2, 1), xp, 'f')
        return xp.append(a, b, axis=0)
