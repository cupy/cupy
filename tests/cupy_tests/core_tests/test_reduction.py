import unittest

import six

import cupy
from cupy import core
from cupy import testing


@testing.gpu
class SimpleReductionFunction(unittest.TestCase):

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
        size = ((2 << 32) // self.my_int8_sum._block_size)
        self.check_int8_sum((size, 1), axis=1)
        self.check_int8_sum((size, 1), axis=0)


@testing.gpu
@testing.parameterize(
    {'use_special_variable_in_map_expr': True},
    {'use_special_variable_in_map_expr': False},
)
class TestReductionKernel(unittest.TestCase):

    def setUp(self):
        if self.use_special_variable_in_map_expr:
            self.my_sum = core.ReductionKernel(
                'T x', 'T out', 'a = x', 'a + b', 'out = a', '0', 'my_sum',
                use_special_variable_in_map_expr=True)
        else:
            self.my_sum = core.ReductionKernel(
                'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum',
                use_special_variable_in_map_expr=False)

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


@testing.gpu
@testing.parameterize(
    {'use_special_variable_in_map_expr': True},
    {'use_special_variable_in_map_expr': False},
)
class TestReductionKernelInvalidArgument(unittest.TestCase):

    def test_invalid_kernel_name(self):
        with six.assertRaisesRegex(self, ValueError, 'Invalid kernel name'):
            if self.use_special_variable_in_map_expr:
                core.ReductionKernel(
                    'T x', 'T y', 'a = x', 'a + b', 'y = a', '0', name='1',
                    use_special_variable_in_map_expr=True)
            else:
                core.ReductionKernel(
                    'T x', 'T y', 'x', 'a + b', 'y = a', '0', name='1',
                    use_special_variable_in_map_expr=False)
