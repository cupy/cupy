import unittest

import six

import cupy
from cupy import core
from cupy import testing


@testing.gpu
class SimpleReductionFunction(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.my_int8_sum = core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None))

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False):
        a = testing.shaped_random(shape, xp, 'b')
        if xp == cupy:
            return self.my_int8_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape1(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i,))
            self.check_int8_sum((2 ** i - 1,))
            self.check_int8_sum((2 ** i + 1,))

    def test_shape2(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=0)
            self.check_int8_sum((2 ** i - 1, 1000), axis=0)
            self.check_int8_sum((2 ** i + 1, 1000), axis=0)

    def test_shape3(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=1)
            self.check_int8_sum((2 ** i - 1, 1000), axis=1)
            self.check_int8_sum((2 ** i + 1, 1000), axis=1)

    def test_shape4(self):
        self.check_int8_sum((512, 256 * 256), axis=0)
        self.check_int8_sum((512, 256 * 256), axis=1)

        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=0)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=1)

    def test_shape5(self):
        size = ((2 << 32) //
                cupy.core.core.simple_reduction_function._block_size)
        self.check_int8_sum((size, 1), axis=1)
        self.check_int8_sum((size, 1), axis=0)


@testing.gpu
class TestReductionKernel(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.my_sum = core.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False):
        a = testing.shaped_random(shape, xp, 'b')
        if xp == cupy:
            return self.my_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape1(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i,))
            self.check_int8_sum((2 ** i - 1,))
            self.check_int8_sum((2 ** i + 1,))

    def test_shape2(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=0)
            self.check_int8_sum((2 ** i - 1, 1000), axis=0)
            self.check_int8_sum((2 ** i + 1, 1000), axis=0)

    def test_shape3(self):
        for i in six.moves.range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=1)
            self.check_int8_sum((2 ** i - 1, 1000), axis=1)
            self.check_int8_sum((2 ** i + 1, 1000), axis=1)

    def test_shape4(self):
        self.check_int8_sum((512, 256 * 256), axis=0)
        self.check_int8_sum((512, 256 * 256), axis=1)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=0)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=1)
