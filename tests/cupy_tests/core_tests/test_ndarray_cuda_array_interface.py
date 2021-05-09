import unittest
import pytest

from cupy_backends.cuda import stream as stream_module
import cupy
from cupy import _core
from cupy import testing


# TODO(leofang): test PTDS in this file


class DummyObjectWithCudaArrayInterface(object):

    def __init__(self, a, ver=3):
        self.a = a
        self.ver = ver

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.a.shape,
            'strides': self.a.strides,
            'typestr': self.a.dtype.str,
            'descr': self.a.dtype.descr,
            'data': (self.a.data.ptr, False),
            'version': self.ver,
        }
        if self.ver == 3:
            stream = cupy.cuda.get_current_stream()
            desc['stream'] = 1 if stream.ptr == 0 else stream.ptr
        return desc


@testing.parameterize(*testing.product({
    'stream': ('null', 'new'),
    'ver': (2, 3),
}))
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestArrayUfunc(unittest.TestCase):

    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError,
                                 contiguous_check=False)
    def check_array_scalar_op(self, op, xp, x_type, y_type, trans=False):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if trans:
            a = a.T

        if xp is cupy:
            with self.stream:
                a = DummyObjectWithCudaArrayInterface(a, self.ver)
                return getattr(xp, op)(a, y_type(3))
        else:
            return getattr(xp, op)(a, y_type(3))

    def test_add_scalar(self):
        self.check_array_scalar_op('add')

    def test_add_scalar_with_strides(self):
        self.check_array_scalar_op('add', trans=True)


@testing.parameterize(*testing.product({
    'stream': ('null', 'new'),
    'ver': (2, 3),
}))
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestElementwiseKernel(unittest.TestCase):

    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()

    @testing.for_all_dtypes_combination()
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError,
                                 contiguous_check=False)
    def check_array_scalar_op(self, op, xp, dtyes, trans=False):
        a = xp.array([[1, 2, 3], [4, 5, 6]], dtyes)
        if trans:
            a = a.T

        if xp is cupy:
            with self.stream:
                a = DummyObjectWithCudaArrayInterface(a, self.ver)
                f = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
                return f(a, dtyes(3))
        else:
            return a + dtyes(3)

    def test_add_scalar(self):
        self.check_array_scalar_op('add')

    def test_add_scalar_with_strides(self):
        self.check_array_scalar_op('add', trans=True)


@testing.parameterize(*testing.product({
    'stream': ('null', 'new'),
    'ver': (2, 3),
}))
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestSimpleReductionFunction(unittest.TestCase):

    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()

        self.my_int8_sum = _core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None))

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False,
                       trans=False):
        a = testing.shaped_random(shape, xp, 'b')
        if trans:
            a = a.T

        if xp == cupy:
            with self.stream:
                a = DummyObjectWithCudaArrayInterface(a, self.ver)
                return self.my_int8_sum(
                    a, axis=axis, keepdims=keepdims)
        else:
            return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def test_shape(self):
        self.check_int8_sum((2 ** 10,))

    def test_shape_with_strides(self):
        self.check_int8_sum((2 ** 10, 16), trans=True)


@testing.parameterize(*testing.product({
    'stream': ('null', 'new'),
    'ver': (2, 3),
}))
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestReductionKernel(unittest.TestCase):

    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()

        self.my_sum = _core.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')

    @testing.numpy_cupy_allclose()
    def check_int8_sum(self, shape, xp, axis=None, keepdims=False,
                       trans=False):
        a = testing.shaped_random(shape, xp, 'b')
        if trans:
            a = a.T

        if xp == cupy:
            with self.stream:
                a = DummyObjectWithCudaArrayInterface(a, self.ver)
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
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
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


test_cases = [
    {'shape': (10,), 'slices': (slice(0, None),)},
    {'shape': (10,), 'slices': (slice(2, None),)},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(0, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(0, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(2, None))},
    {'shape': (10, 10), 'slices': (slice(2, None), slice(4, None))},
]
test_streams = ('null', 'new')
test_cases_with_stream = [
    {'stream': s, **t} for t in test_cases for s in test_streams]


@testing.parameterize(*test_cases_with_stream)
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestCUDAArrayInterfaceCompliance(unittest.TestCase):

    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()

    @testing.for_all_dtypes_combination(names=['dtype'])
    @testing.for_orders('CF')
    def test_value_type(self, dtype, order):
        x = cupy.zeros(self.shape, dtype=dtype, order=order)
        y = x[self.slices]

        # mandatory entries
        with self.stream:
            CAI = y.__cuda_array_interface__
        shape = CAI['shape']
        typestr = CAI['typestr']
        ptr, readonly = CAI['data']
        version = CAI['version']
        strides = CAI['strides']

        # optional entries
        descr = CAI['descr'] if 'descr' in CAI else None
        stream = CAI['stream'] if 'stream' in CAI else None

        # Don't validate correctness of data here, just their types
        assert version == 3  # bump this when the protocol is updated!
        assert isinstance(CAI, dict)
        assert isinstance(shape, tuple)
        assert isinstance(typestr, str)
        assert isinstance(ptr, int)
        assert isinstance(readonly, bool)
        assert (strides is None) or isinstance(strides, tuple)
        assert (descr is None) or isinstance(descr, list)
        if isinstance(descr, list):
            for item in descr:
                assert isinstance(item, tuple)
        assert (stream is None) or isinstance(stream, int)


@testing.parameterize(*testing.product({
    'stream': ('null', 'new', 'ptds'),
}))
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestCUDAArrayInterfaceStream(unittest.TestCase):
    def setUp(self):
        if self.stream == 'null':
            self.stream = cupy.cuda.Stream.null
        elif self.stream == 'new':
            self.stream = cupy.cuda.Stream()
        elif self.stream == 'ptds':
            self.stream = cupy.cuda.Stream.ptds

    def test_stream_export(self):
        a = cupy.empty(100)

        # the stream context should export the stream
        with self.stream:
            stream_ptr = a.__cuda_array_interface__['stream']

        if self.stream is cupy.cuda.Stream.null:
            assert stream_ptr == stream_module.get_default_stream_ptr()
        elif self.stream is cupy.cuda.Stream.ptds:
            assert stream_ptr == 2
        else:
            assert stream_ptr == self.stream.ptr

        # without a stream context, it's always the default stream
        stream_ptr = a.__cuda_array_interface__['stream']
        assert stream_ptr == stream_module.get_default_stream_ptr()
