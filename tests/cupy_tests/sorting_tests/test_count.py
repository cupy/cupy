import unittest

import numpy
import six

import cupy
from cupy import testing


@testing.gpu
class TestCount(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_count_nonzero(self, dtype):
        def func(xp):
            m = testing.shaped_random((2, 3), xp, xp.bool_)
            a = testing.shaped_random((2, 3), xp, dtype) * m
            c = xp.count_nonzero(a)
            self.assertIsInstance(c, int)
            return c
        self.assertEqual(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_count_nonzero_zero_dim(self, dtype):
        def func(xp):
            a = xp.array(1.0, dtype=dtype)
            c = xp.count_nonzero(a)
            self.assertIsInstance(c, int)
            return c
        self.assertEqual(func(numpy), func(cupy))

    @testing.with_requires('numpy>=1.12')
    @testing.for_all_dtypes()
    def test_count_nonzero_int_axis(self, dtype):
        for ax in six.moves.range(3):
            def func(xp):
                m = testing.shaped_random((2, 3, 4), xp, xp.bool_)
                a = testing.shaped_random((2, 3, 4), xp, dtype) * m
                return xp.count_nonzero(a, axis=ax)
            testing.assert_allclose(func(numpy), func(cupy))

    @testing.with_requires('numpy>=1.12')
    @testing.for_all_dtypes()
    def test_count_nonzero_tuple_axis(self, dtype):
        for ax in six.moves.range(3):
            for ay in six.moves.range(3):
                if ax == ay:
                    continue

                def func(xp):
                    m = testing.shaped_random((2, 3, 4), xp, xp.bool_)
                    a = testing.shaped_random((2, 3, 4), xp, dtype) * m
                    return xp.count_nonzero(a, axis=(ax, ay))
                testing.assert_allclose(func(numpy), func(cupy))
