import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestShape(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_reshape_strides(self):
        def func(xp):
            a = testing.shaped_arange((1, 1, 1, 2, 2), xp)
            return a.strides
        self.assertEqual(func(numpy), func(cupy))

    def test_reshape2(self):
        def func(xp):
            a = xp.zeros((8,), dtype=xp.float32)
            return a.reshape((1, 1, 1, 4, 1, 2)).strides
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nocopy_reshape(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2)
        b[1] = 1
        return a

    @testing.numpy_cupy_array_equal()
    def test_transposed_reshape(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).T
        return a.reshape(4, 6)

    @testing.numpy_cupy_array_equal()
    def test_transposed_reshape2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.reshape(2, 3, 4)

    @testing.numpy_cupy_array_equal()
    def test_reshape_with_unknown_dimension(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.reshape(3, -1)

    @testing.numpy_cupy_raises()
    def test_reshape_with_multiple_unknown_dimensions(self):
        a = testing.shaped_arange((2, 3, 4))
        a.reshape(3, -1, -1)

    @testing.numpy_cupy_raises()
    def test_reshape_with_changed_arraysize(self):
        a = testing.shaped_arange((2, 3, 4))
        a.reshape(2, 4, 4)

    @testing.numpy_cupy_array_equal()
    def test_external_reshape(self, xp):
        a = xp.zeros((8,), dtype=xp.float32)
        return xp.reshape(a, (1, 1, 1, 4, 1, 2))

    @testing.numpy_cupy_array_equal()
    def test_ravel(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return a.ravel()

    @testing.numpy_cupy_array_equal()
    def test_external_ravel(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return xp.ravel(a)

    def test_reshape_contiguity(self):
        a = cupy.arange(6).reshape(2, 3)
        self.assertTrue(a.flags.c_contiguous)
        self.assertFalse(a.flags.f_contiguous)

        a = a.reshape(1, 6, 1)
        self.assertTrue(a.flags.c_contiguous)
        self.assertTrue(a.flags.f_contiguous)

        b = a.T.reshape(1, 6, 1)
        self.assertTrue(b.flags.c_contiguous)
        self.assertTrue(b.flags.f_contiguous)

        b = a.T.reshape(2, 3)
        self.assertTrue(b.flags.c_contiguous)
        self.assertFalse(b.flags.f_contiguous)
