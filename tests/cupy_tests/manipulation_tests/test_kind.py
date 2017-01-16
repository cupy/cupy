import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestKind(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_asfortranarray1(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype)
            ret = xp.asfortranarray(x)
            self.assertTrue(x.flags.c_contiguous)
            self.assertTrue(ret.flags.f_contiguous)
            return ret.strides
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_asfortranarray2(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype)
            ret = xp.asfortranarray(x)
            self.assertTrue(x.flags.c_contiguous)
            self.assertTrue(ret.flags.f_contiguous)
            return ret.strides
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_asfortranarray3(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype)
            ret = xp.asfortranarray(xp.asfortranarray(x))
            self.assertTrue(x.flags.c_contiguous)
            self.assertTrue(ret.flags.f_contiguous)
            return ret.strides
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_asfortranarray4(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype)
            x = xp.transpose(x, (1, 0))
            ret = xp.asfortranarray(x)
            self.assertTrue(ret.flags.f_contiguous)
            return ret.strides
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_asfortranarray5(self, dtype):
        def func(xp):
            x = testing.shaped_arange((2, 3), xp, dtype)
            ret = xp.asfortranarray(x)
            self.assertTrue(x.flags.c_contiguous)
            self.assertTrue(ret.flags.f_contiguous)
            return ret.strides
        self.assertEqual(func(numpy), func(cupy))
