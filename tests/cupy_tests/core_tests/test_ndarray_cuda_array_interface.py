import unittest

import cupy
from cupy import core
from cupy import testing


class DummyObjectWithCudaArrayInterface(object):

    def __init__(self, a):
        self.a = a

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.a.shape,
            'strides': self.a.strides,
            'typestr': self.a.dtype.str,
            'descr': self.a.dtype.descr,
            'data': (self.a.data.mem.ptr, False),
            'version': 0,
        }
        return desc


@testing.gpu
class TestArrayUfunc(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError,
                                 contiguous_check=False)
    def check_array_scalar_op(self, op, xp, x_type, y_type, trans=False):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if trans:
            a = a.T

        if xp is cupy:
            a = DummyObjectWithCudaArrayInterface(a)
        return getattr(xp, op)(a, y_type(3))

    def test_add_scalar(self):
        self.check_array_scalar_op('add')

    def test_add_scalar_with_strides(self):
        self.check_array_scalar_op('add', trans=True)


@testing.gpu
class TestElementwiseKernel(unittest.TestCase):

    @testing.for_all_dtypes_combination()
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError,
                                 contiguous_check=False)
    def check_array_scalar_op(self, op, xp, dtyes, trans=False):
        a = xp.array([[1, 2, 3], [4, 5, 6]], dtyes)
        if trans:
            a = a.T

        if xp is cupy:
            a = DummyObjectWithCudaArrayInterface(a)
            f = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
            return f(a, dtyes(3))
        else:
            return a + dtyes(3)

    def test_add_scalar(self):
        self.check_array_scalar_op('add')

    def test_add_scalar_with_strides(self):
        self.check_array_scalar_op('add', trans=True)


@testing.gpu
class SimpleReductionFunction(unittest.TestCase):

    def setUp(self):
        self.my_int8_sum = core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None))

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False,
                       trans=False):
        a = testing.shaped_random(shape, xp, 'b')
        if trans:
            a = a.T

        if xp == cupy:
            a = DummyObjectWithCudaArrayInterface(a)
            return self.my_int8_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape(self):
        self.check_int8_sum((2 ** 10,))

    def test_shape_with_strides(self):
        self.check_int8_sum((2 ** 10, 16), trans=True)


@testing.gpu
class TestReductionKernel(unittest.TestCase):

    def setUp(self):
        self.my_sum = core.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False,
                       trans=False):
        a = testing.shaped_random(shape, xp, 'b')
        if trans:
            a = a.T

        if xp == cupy:
            a = DummyObjectWithCudaArrayInterface(a)
            return self.my_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape(self):
        self.check_int8_sum((2 ** 10,))

    def test_shape_with_strides(self):
        self.check_int8_sum((2 ** 10, 16), trans=True)
