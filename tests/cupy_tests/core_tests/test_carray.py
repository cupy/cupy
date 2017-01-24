import unittest

import cupy
from cupy import testing


class TestCArray(unittest.TestCase):

    def test_size(self):
        x = cupy.arange(3).astype('i')
        y = cupy.ElementwiseKernel(
            'raw int32 x', 'int32 y', 'y = x.size()', 'test_carray_size',
        )(x, size=1)
        self.assertEqual(int(y[0]), 3)

    def test_shape(self):
        x = cupy.arange(6).reshape((2, 3)).astype('i')
        y = cupy.ElementwiseKernel(
            'raw int32 x', 'int32 y', 'y = x.shape()[i]', 'test_carray_shape',
        )(x, size=2)
        testing.assert_array_equal(y, (2, 3))

    def test_strides(self):
        x = cupy.arange(6).reshape((2, 3)).astype('i')
        y = cupy.ElementwiseKernel(
            'raw int32 x', 'int32 y', 'y = x.strides()[i]',
            'test_carray_strides',
        )(x, size=2)
        testing.assert_array_equal(y, (12, 4))

    def test_getitem_int(self):
        x = cupy.arange(24).reshape((2, 3, 4)).astype('i')
        y = cupy.empty_like(x)
        y = cupy.ElementwiseKernel(
            'raw T x', 'int32 y', 'y = x[i]', 'test_carray_getitem_int',
        )(x, y)
        testing.assert_array_equal(y, x)

    def test_getitem_idx(self):
        x = cupy.arange(24).reshape((2, 3, 4)).astype('i')
        y = cupy.empty_like(x)
        y = cupy.ElementwiseKernel(
            'raw T x', 'int32 y',
            'int idx[] = {i / 12, i / 4 % 3, i % 4}; y = x[idx]',
            'test_carray_getitem_idx',
        )(x, y)
        testing.assert_array_equal(y, x)
