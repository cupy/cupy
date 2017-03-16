import unittest

import cupy

from cupy import testing


@testing.gpu
class TestJoin(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_column_stack(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((4, 3), xp, dtype1)
        b = testing.shaped_arange((4,), xp, dtype2)
        c = testing.shaped_arange((4, 2), xp, dtype1)
        return xp.column_stack((a, b, c))

    def test_column_stack_wrong_ndim1(self):
        a = cupy.zeros(())
        b = cupy.zeros((3,))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_ndim2(self):
        a = cupy.zeros((3, 2, 3))
        b = cupy.zeros((3, 2))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_shape(self):
        a = cupy.zeros((3, 2))
        b = cupy.zeros((4, 3))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=2)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 5), xp, dtype)
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 2), xp, dtype).T
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e), axis=-1)

    def test_concatenate_wrong_ndim(self):
        a = cupy.empty((2, 3))
        b = cupy.empty((2,))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b))

    def test_concatenate_wrong_shape(self):
        a = cupy.empty((2, 3, 4))
        b = cupy.empty((3, 3, 4))
        c = cupy.empty((4, 4, 4))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_dstack(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        b = testing.shaped_arange((3,), xp)
        c = testing.shaped_arange((1, 3), xp)
        return xp.dstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element(self, xp):
        a = testing.shaped_arange((1, 2, 3), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element_2(self, xp):
        a = testing.shaped_arange((1, 2), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element_3(self, xp):
        a = testing.shaped_arange((1,), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_hstack_vectors(self, xp):
        a = xp.arange(3)
        b = xp.arange(2, -1, -1)
        return xp.hstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_hstack_scalars(self, xp):
        a = testing.shaped_arange((), xp)
        b = testing.shaped_arange((), xp)
        c = testing.shaped_arange((), xp)
        return xp.hstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_hstack(self, xp):
        a = testing.shaped_arange((2, 1), xp)
        b = testing.shaped_arange((2, 2), xp)
        c = testing.shaped_arange((2, 3), xp)
        return xp.hstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_vstack_vectors(self, xp):
        a = xp.arange(3)
        b = xp.arange(2, -1, -1)
        return xp.vstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_vstack_single_element(self, xp):
        a = xp.arange(3)
        return xp.vstack((a,))

    def test_vstack_wrong_ndim(self):
        a = cupy.empty((3,))
        b = cupy.empty((3, 1))
        with self.assertRaises(ValueError):
            cupy.vstack((a, b))

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_array_equal()
    def test_stack(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        b = testing.shaped_arange((2, 3), xp)
        c = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, b, c))

    def test_stack_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        b = testing.shaped_arange((2, 3), cupy)
        c = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, b, c))
        self.assertEqual(s.shape, (3, 2, 3))
        cupy.testing.assert_array_equal(s[0], a)
        cupy.testing.assert_array_equal(s[1], b)
        cupy.testing.assert_array_equal(s[2], c)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_array_equal()
    def test_stack_with_axis(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, a), axis=1)

    def test_stack_with_axis_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, a), axis=1)

        self.assertEqual(s.shape, (2, 2, 3))
        cupy.testing.assert_array_equal(s[:, 0, :], a)
        cupy.testing.assert_array_equal(s[:, 1, :], a)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_array_equal()
    def test_stack_with_negative_axis(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, a), axis=-1)

    def test_stack_with_negative_axis_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, a), axis=-1)

        self.assertEqual(s.shape, (2, 3, 2))
        cupy.testing.assert_array_equal(s[:, :, 0], a)
        cupy.testing.assert_array_equal(s[:, :, 1], a)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_raises()
    def test_stack_different_shape(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        b = testing.shaped_arange((2, 4), xp)
        return xp.stack([a, b])

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_raises()
    def test_stack_out_of_bounds(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack([a, a], axis=3)
