import operator
import unittest

import numpy
import six

import cupy
from cupy import core
from cupy import testing


class DummyObjectWithCudaArrayInterface():

    def __init__(self, a):
        self.a = a

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.a.shape,
            'typestr': self.a.dtype.str,
            'descr': self.a.dtype.descr,
            'data': (self.a.data.mem.ptr, False),
            'version': 0,
        }
        return desc


@testing.gpu
class TestArrayUfunc(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False,
                              no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if xp is cupy:
            a = DummyObjectWithCudaArrayInterface(a)
        if swap:
            return getattr(xp, op)(y_type(3), a)
        else:
            return getattr(xp, op)(a, y_type(3))

    def test_add_scalar(self):
        self.check_array_scalar_op('add')


@testing.gpu
class SimpleReductionFunction(unittest.TestCase):

    def setUp(self):
        self.my_int8_sum = core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None))

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False):
        a = testing.shaped_random(shape, xp, 'b')
        if xp == cupy:
            a = DummyObjectWithCudaArrayInterface(a)
            return self.my_int8_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape(self):
        self.check_int8_sum((2 ** 10,))


@testing.gpu
class TestReductionKernel(unittest.TestCase):

    def setUp(self):
        self.my_sum = core.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False):
        a = testing.shaped_random(shape, xp, 'b')
        if xp == cupy:
            a = DummyObjectWithCudaArrayInterface(a)
            return self.my_sum(
                a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape(self):
        self.check_int8_sum((2 ** 10,))
