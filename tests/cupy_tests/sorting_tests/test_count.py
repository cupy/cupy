import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestCount(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_count_nonzero(self, dtype):
        def func(xp):
            m = testing.shaped_random((2, 3), xp, xp.bool_)
            a = testing.shaped_random((2, 3), xp, dtype) * m
            c = xp.count_nonzero(a)
            if xp is cupy:
                # CuPy returns zero-dimensional array instead of
                # returning a scalar value
                assert isinstance(c, xp.ndarray)
                assert c.dtype == 'l'
                assert c.shape == ()
            return int(c)
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_count_nonzero_zero_dim(self, dtype):
        def func(xp):
            a = xp.array(1.0, dtype=dtype)
            c = xp.count_nonzero(a)
            if xp is cupy:
                # CuPy returns zero-dimensional array instead of
                # returning a scalar value
                assert isinstance(c, xp.ndarray)
                assert c.dtype == 'l'
                assert c.shape == ()
            return int(c)
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_count_nonzero_int_axis(self, dtype):
        for ax in range(3):
            def func(xp):
                m = testing.shaped_random((2, 3, 4), xp, xp.bool_)
                a = testing.shaped_random((2, 3, 4), xp, dtype) * m
                return xp.count_nonzero(a, axis=ax)
            testing.assert_allclose(func(numpy), func(cupy))

    @testing.for_all_dtypes()
    def test_count_nonzero_tuple_axis(self, dtype):
        for ax in range(3):
            for ay in range(3):
                if ax == ay:
                    continue

                def func(xp):
                    m = testing.shaped_random((2, 3, 4), xp, xp.bool_)
                    a = testing.shaped_random((2, 3, 4), xp, dtype) * m
                    return xp.count_nonzero(a, axis=(ax, ay))
                testing.assert_allclose(func(numpy), func(cupy))
