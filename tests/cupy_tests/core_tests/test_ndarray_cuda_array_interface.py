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
            'data': (self.a.data.ptr, False),
            'version': 2,
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


@testing.parameterize(
    {'shape': (10,), 'slices': (slice(0, None),)},
    {'shape': (10,), 'slices': (slice(2, None),)},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(4, None))},
)
@testing.gpu
class TestSlicingMemoryPointer(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['dtype'])
    @testing.for_orders('CF')
    def test_shape_with_strides(self, dtype, order):
        x = cupy.zeros(self.shape, dtype=dtype, order=order)

        start = [s.start for s in self.slices]
        itemsize = cupy.dtype(dtype).itemsize
        dimsize = [s * itemsize for s in start]
        if len(self.shape) == 1:
            offset = start[0] * itemsize
        else:
            if order == 'C':
                offset = self.shape[0] * dimsize[0] + dimsize[1]
            else:
                offset = self.shape[0] * dimsize[1] + dimsize[0]

        cai_ptr, _ = x.__cuda_array_interface__['data']
        slice_cai_ptr, _ = x[self.slices].__cuda_array_interface__['data']
        cupy_data_ptr = x.data.ptr
        sliced_cupy_data_ptr = x[self.slices].data.ptr

        assert cai_ptr == cupy_data_ptr
        assert slice_cai_ptr == sliced_cupy_data_ptr
        assert slice_cai_ptr == cai_ptr+offset


@testing.parameterize(
    {'shape': (10,), 'slices': (slice(0, None),)},
    {'shape': (10,), 'slices': (slice(2, None),)},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(4, None))},
)
@testing.gpu
class TestCUDAArrayInterfaceCompliance(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['dtype'])
    @testing.for_orders('CF')
    def test_value_type(self, dtype, order):
        x = cupy.zeros(self.shape, dtype=dtype, order=order)
        y = x[self.slices]

        # mandatory entries
        shape = y.__cuda_array_interface__['shape']
        typestr = y.__cuda_array_interface__['typestr']
        ptr, readonly = y.__cuda_array_interface__['data']
        version = y.__cuda_array_interface__['version']
        strides = y.__cuda_array_interface__['strides']

        # optional entries
        if 'descr' in y.__cuda_array_interface__:
            descr = y.__cuda_array_interface__['descr']
        else:
            descr = None

        # Don't validate correctness of data here, just their types
        assert isinstance(shape, tuple)
        assert isinstance(typestr, str)
        assert isinstance(ptr, int)
        assert isinstance(readonly, bool)
        assert version == 2  # update this when the standard is updated!
        assert (strides is None) or isinstance(strides, tuple)
        assert (descr is None) or isinstance(descr, list)
        if isinstance(descr, list):
            for item in descr:
                assert isinstance(item, tuple)
