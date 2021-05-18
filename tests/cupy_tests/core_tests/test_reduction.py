import unittest

import numpy
import pytest

import cupy
from cupy import _core
from cupy import testing


_noncontiguous_params = [
    # reduce at head axes
    {'shape': (2, 4, 3), 'trans': (2, 1, 0), 'axis': (0, 1)},
    # reduce at middle axes
    {'shape': (2, 4, 5, 3), 'trans': (3, 2, 1, 0), 'axis': (1, 2)},
    # reduce at tail axes
    {'shape': (2, 4, 3), 'trans': (2, 1, 0), 'axis': (1, 2)},
    # out_axis = (0,)
    {'shape': (0, 4, 3), 'trans': (2, 1, 0), 'axis': (0, 1)},
    # out_axis = ()
    {'shape': (2, 4, 3), 'trans': (2, 1, 0), 'axis': (0, 1, 2)},
]


class AbstractReductionTestBase:

    def get_sum_func(self):
        raise NotImplementedError()

    @testing.numpy_cupy_allclose(contiguous_check=False)
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False, trans=None):
        a = testing.shaped_random(shape, xp, 'b')
        if trans:
            a = a.transpose(*trans)
        sum_func = self.get_sum_func()
        if xp == cupy:
            return sum_func(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')


class SimpleReductionFunctionTestBase(AbstractReductionTestBase):

    def get_sum_func(self):
        return _core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None), 0)


@testing.gpu
class TestSimpleReductionFunction(
        unittest.TestCase, SimpleReductionFunctionTestBase):
    def test_shape1(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i,))
            self.check_int8_sum((2 ** i - 1,))
            self.check_int8_sum((2 ** i + 1,))

    def test_shape2(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=0)
            self.check_int8_sum((2 ** i - 1, 1000), axis=0)
            self.check_int8_sum((2 ** i + 1, 1000), axis=0)

    def test_shape3(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=1)
            self.check_int8_sum((2 ** i - 1, 1000), axis=1)
            self.check_int8_sum((2 ** i + 1, 1000), axis=1)

    def test_shape4(self):
        self.check_int8_sum((512, 256 * 256), axis=0)
        self.check_int8_sum((512, 256 * 256), axis=1)

        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=0)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=1)

    def test_shape5(self):
        block_size = 512
        size = ((2 << 32) // block_size)
        self.check_int8_sum((size, 1), axis=1)
        self.check_int8_sum((size, 1), axis=0)


@testing.gpu
@testing.parameterize(*_noncontiguous_params)
class TestSimpleReductionFunctionNonContiguous(
        SimpleReductionFunctionTestBase, unittest.TestCase):

    def test_noncontiguous(self):
        self.check_int8_sum(self.shape, trans=self.trans, axis=self.axis)


@testing.parameterize(*testing.product({
    'backend': ([], ['cub']),
}))
@testing.gpu
class TestSimpleReductionFunctionComplexWarning(unittest.TestCase):

    def setUp(self):
        self.accelerators = _core.get_reduction_accelerators()
        _core.set_reduction_accelerators(self.backend)

    def tearDown(self):
        _core.set_reduction_accelerators(self.accelerators)

    @testing.for_complex_dtypes(name='c_dtype')
    @testing.for_float_dtypes(name='f_dtype')
    @testing.numpy_cupy_allclose()
    def test_warns(self, xp, c_dtype, f_dtype):
        with pytest.warns(numpy.ComplexWarning):
            out = xp.ones((8,), dtype=c_dtype).sum(dtype=f_dtype)
        return out


class ReductionKernelTestBase(AbstractReductionTestBase):

    def get_sum_func(self):
        return cupy.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')


@testing.gpu
class TestReductionKernel(ReductionKernelTestBase, unittest.TestCase):

    def test_shape1(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i,))
            self.check_int8_sum((2 ** i - 1,))
            self.check_int8_sum((2 ** i + 1,))

    def test_shape2(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=0)
            self.check_int8_sum((2 ** i - 1, 1000), axis=0)
            self.check_int8_sum((2 ** i + 1, 1000), axis=0)

    def test_shape3(self):
        for i in range(1, 10):
            self.check_int8_sum((2 ** i, 1000), axis=1)
            self.check_int8_sum((2 ** i - 1, 1000), axis=1)
            self.check_int8_sum((2 ** i + 1, 1000), axis=1)

    def test_shape4(self):
        self.check_int8_sum((512, 256 * 256), axis=0)
        self.check_int8_sum((512, 256 * 256), axis=1)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=0)
        self.check_int8_sum((512 + 1, 256 * 256 + 1), axis=1)


@testing.gpu
@testing.parameterize(*_noncontiguous_params)
class TestReductionKernelNonContiguous(
        ReductionKernelTestBase, unittest.TestCase):

    def test_noncontiguous(self):
        self.check_int8_sum(self.shape, trans=self.trans, axis=self.axis)


@testing.gpu
class TestReductionKernelInvalidArgument(unittest.TestCase):

    def test_invalid_kernel_name(self):
        with self.assertRaisesRegex(ValueError, 'Invalid kernel name'):
            cupy.ReductionKernel(
                'T x', 'T y', 'x', 'a + b', 'y = a', '0', name='1')


@testing.gpu
class TestLargeMultiDimReduction(
        ReductionKernelTestBase, unittest.TestCase):

    def test_large_dims_keep_kernels(self):
        # This test creates a CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
        # if the output array dims are not reduced
        shape = (4, 3, 2, 4, 3, 2, 2)
        axis = (1, 4, 3, 6)
        self.check_int8_sum(shape, axis=axis, keepdims=True)
