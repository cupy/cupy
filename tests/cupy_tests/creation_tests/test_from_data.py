import tempfile
import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing
import numpy


@testing.gpu
class TestFromData(unittest.TestCase):

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array(self, xp, dtype, order):
        return xp.array([[1, 2, 3], [2, 3, 4]], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_empty_list(self, xp, dtype, order):
        return xp.array([], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_nested_empty_list(self, xp, dtype, order):
        return xp.array([[], []], dtype=dtype, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_numpy_scalar(self, xp, dtype, order):
        a = numpy.array(2, dtype=dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_numpy_broad_cast(self, xp, dtype, order):
        a = testing.shaped_arange((2, 1, 4), numpy, dtype)
        a = numpy.broadcast_to(a, (2, 3, 4))
        return xp.array(a, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy(self, xp, dtype, src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy_view(self, xp, dtype, src_order,
                                           dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)

        # create a list of view of ndarrays
        a = [
            (testing.shaped_arange((3, 8), numpy,
                                   dtype, src_order) + (24 * i))[:, ::2]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_numpy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)
        a = [numpy.array(i, dtype=dtype) for i in range(2)]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_nested_list_of_numpy(self, xp, dtype, src_order,
                                             dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)
        a = [
            [testing.shaped_arange(
                (3, 4), numpy, dtype, src_order) + (12 * i)]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_cupy(
            self, xp, dtype1, dtype2, src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype1, src_order),
            testing.shaped_arange((3, 4), xp, dtype2, src_order),
        ]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_cupy_view(self, xp, dtype, src_order,
                                          dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)

        # create a list of view of ndarrays
        a = [
            (testing.shaped_arange((3, 8), xp,
                                   dtype, src_order) + (24 * i))[:, ::2]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_nested_list_of_cupy(self, xp, dtype, src_order,
                                            dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [
            [testing.shaped_arange((3, 4), xp, dtype, src_order) + (12 * i)]
            for i in range(2)]
        return xp.array(a, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_from_list_of_cupy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [xp.array(i, dtype=dtype) for i in range(2)]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_from_nested_list_of_cupy_scalar(self, xp, dtype, order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [[xp.array(i, dtype=dtype)] for i in range(2)]
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.array(a, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy_is_copied(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype(self, xp, dtype1, dtype2, order):
        # complex to real makes no sense
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=dtype2, order=order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype_char(self, xp, dtype1, dtype2, order):
        # complex to real makes no sense
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=order)

    @testing.for_orders('CFAK')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype_being_none(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return xp.array(a, dtype=None, order=order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_copy_list_of_numpy_with_dtype(self, xp, dtype1, dtype2,
                                                 src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=dtype2, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_copy_list_of_numpy_with_dtype_char(self, xp, dtype1,
                                                      dtype2, src_order,
                                                      dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of numpy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), numpy, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_copy_list_of_cupy_with_dtype(self, xp, dtype1, dtype2,
                                                src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=dtype2, order=dst_order)

    @testing.for_orders('CFAK', name='src_order')
    @testing.for_orders('CFAK', name='dst_order')
    @testing.for_all_dtypes(name='dtype1', no_complex=True)
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_array_copy_list_of_cupy_with_dtype_char(self, xp, dtype1, dtype2,
                                                     src_order, dst_order):
        # compares numpy.array(<list of numpy.ndarray>) with
        # cupy.array(<list of cupy.ndarray>)
        a = [
            testing.shaped_arange((3, 4), xp, dtype1, src_order) + (12 * i)
            for i in range(2)]
        return xp.array(a, dtype=numpy.dtype(dtype2).char, order=dst_order)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_no_copy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order=order)
        a.fill(0)
        return b

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_f_contiguous_input(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype, order='F')
        b = xp.array(a, copy=False, order=order)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_f_contiguous_output(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, order='F')
        assert b.flags.f_contiguous
        return b

    @testing.multi_gpu(2)
    def test_array_multi_device(self):
        with cuda.Device(0):
            x = testing.shaped_arange((2, 3, 4), cupy, dtype='f')
        with cuda.Device(1):
            y = cupy.array(x)
        assert isinstance(y, cupy.ndarray)
        assert x is not y  # Do copy
        assert int(x.device) == 0
        assert int(y.device) == 1
        testing.assert_array_equal(x, y)

    @testing.multi_gpu(2)
    def test_array_multi_device_zero_size(self):
        with cuda.Device(0):
            x = testing.shaped_arange((0,), cupy, dtype='f')
        with cuda.Device(1):
            y = cupy.array(x)
        assert isinstance(y, cupy.ndarray)
        assert x is not y  # Do copy
        assert x.device.id == 0
        assert y.device.id == 1
        testing.assert_array_equal(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_no_copy_ndmin(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a, copy=False, ndmin=5)
        assert a.shape == (2, 3, 4)
        a.fill(0)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.asarray(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_is_not_copied(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a)
        a.fill(0)
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_preserves_numpy_array_order(self, xp, dtype, order):
        a_numpy = testing.shaped_arange((2, 3, 4), numpy, dtype, order)
        b = xp.asarray(a_numpy)
        assert b.flags.f_contiguous == a_numpy.flags.f_contiguous
        assert b.flags.c_contiguous == a_numpy.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asanyarray_with_order(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asanyarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_from_numpy(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        b = xp.asarray(a, order=order)
        if order in ['F', 'f']:
            assert b.flags.f_contiguous
        else:
            assert b.flags.c_contiguous
        return b

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_with_order_copy_behavior(self, xp, dtype, order):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a, order=order)
        a.fill(0)
        return b

    def test_ascontiguousarray_on_noncontiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = a.transpose(2, 0, 1)
        c = cupy.ascontiguousarray(b)
        assert c.flags.c_contiguous
        testing.assert_array_equal(b, c)

    def test_ascontiguousarray_on_contiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = cupy.ascontiguousarray(a)
        assert a is b

    @testing.numpy_cupy_array_equal()
    def test_asarray_cuda_array_zero_dim(self, xp):
        a = xp.ones(())
        return xp.ascontiguousarray(a)

    @testing.numpy_cupy_array_equal()
    def test_asarray_cuda_array_zero_dim_dtype(self, xp):
        a = xp.ones((), dtype=numpy.float64)
        return xp.ascontiguousarray(a, dtype=numpy.int64)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copy(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = xp.copy(a, order=order)
        a[1] = 1
        return b

    @testing.multi_gpu(2)
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_copy_multigpu(self, dtype, order):
        with cuda.Device(0):
            src = cupy.random.uniform(-1, 1, (2, 3)).astype(dtype)
        with cuda.Device(1):
            dst = cupy.copy(src, order)
        testing.assert_allclose(src, dst, rtol=0, atol=0)

    @testing.for_CF_orders()
    @testing.numpy_cupy_equal()
    def test_copy_order(self, xp, order):
        a = xp.zeros((2, 3, 4), order=order)
        b = xp.copy(a)
        return (b.flags.c_contiguous, b.flags.f_contiguous)

    @testing.numpy_cupy_array_equal()
    def test_asfortranarray_cuda_array_zero_dim(self, xp):
        a = xp.ones(())
        return xp.asfortranarray(a)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'],
                                        no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_asfortranarray_cuda_array_zero_dim_dtype(
            self, xp, dtype_a, dtype_b):
        a = xp.ones((), dtype=dtype_a)
        return xp.asfortranarray(a, dtype=dtype_b)

    @testing.numpy_cupy_array_equal()
    def test_fromfile(self, xp):
        with tempfile.TemporaryFile() as fh:
            fh.write(b"\x00\x01\x02\x03\x04")
            fh.flush()
            fh.seek(0)
            return xp.fromfile(fh, dtype="u1")


max_cuda_array_interface_version = 3


@testing.gpu
@testing.parameterize(*testing.product({
    'ver': tuple(range(max_cuda_array_interface_version+1)),
    'strides': (False, None, True),
}))
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='HIP does not support this')
class TestCudaArrayInterface(unittest.TestCase):
    @testing.for_all_dtypes()
    def test_base(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(
            DummyObjectWithCudaArrayInterface(a, self.ver, self.strides))
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_not_copied(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(
            DummyObjectWithCudaArrayInterface(a, self.ver, self.strides))
        a.fill(0)
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_order(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = cupy.asarray(
            DummyObjectWithCudaArrayInterface(a, self.ver, self.strides),
            order='F')
        assert b.flags.f_contiguous
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_with_strides(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype).T
        b = cupy.asarray(
            DummyObjectWithCudaArrayInterface(a, self.ver, self.strides))
        assert a.strides == b.strides
        assert a.nbytes == b.data.mem.size

    @testing.for_all_dtypes()
    def test_with_zero_size_array(self, dtype):
        a = testing.shaped_arange((0,), cupy, dtype)
        b = cupy.asarray(
            DummyObjectWithCudaArrayInterface(a, self.ver, self.strides))
        assert a.strides == b.strides
        assert a.nbytes == b.data.mem.size
        assert a.data.ptr == 0
        assert a.size == 0

    @testing.for_all_dtypes()
    def test_asnumpy(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        b = DummyObjectWithCudaArrayInterface(a, self.ver, self.strides)
        a_cpu = cupy.asnumpy(a)
        b_cpu = cupy.asnumpy(b)
        testing.assert_array_equal(a_cpu, b_cpu)


@testing.gpu
@testing.parameterize(*testing.product({
    'ver': tuple(range(1, max_cuda_array_interface_version+1)),
    'strides': (False, None, True),
}))
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='HIP does not support this')
class TestCudaArrayInterfaceMaskedArray(unittest.TestCase):
    # TODO(leofang): update this test when masked array is supported
    @testing.for_all_dtypes()
    def test_masked_array(self, dtype):
        a = testing.shaped_arange((2, 3, 4), cupy, dtype)
        mask = testing.shaped_arange((2, 3, 4), cupy, dtype)
        a = DummyObjectWithCudaArrayInterface(a, self.ver, self.strides, mask)
        with pytest.raises(ValueError) as ex:
            b = cupy.asarray(a)  # noqa
        assert 'does not support' in str(ex.value)


# marked slow as either numpy or cupy could go OOM in this test
@testing.slow
@testing.gpu
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='HIP does not support this')
class TestCudaArrayInterfaceBigArray(unittest.TestCase):
    def test_with_over_size_array(self):
        # real example from #3009
        size = 5 * 10**8
        a = testing.shaped_random((size,), cupy, cupy.float64)
        b = cupy.asarray(DummyObjectWithCudaArrayInterface(a, 2, None))
        testing.assert_array_equal(a, b)


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='HIP does not support this')
class DummyObjectWithCudaArrayInterface(object):
    def __init__(self, a, ver, include_strides=False, mask=None, stream=None):
        assert ver in tuple(range(max_cuda_array_interface_version+1))
        self.a = a
        self.ver = ver
        self.include_strides = include_strides
        self.mask = mask
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.a.shape,
            'typestr': self.a.dtype.str,
            'descr': self.a.dtype.descr,
            'data': (self.a.data.ptr, False),
            'version': self.ver,
        }
        if self.a.flags.c_contiguous:
            if self.include_strides is True:
                desc['strides'] = self.a.strides
            elif self.include_strides is None:
                desc['strides'] = None
            else:  # self.include_strides is False
                pass
        else:  # F contiguous or neither
            desc['strides'] = self.a.strides
        if self.mask is not None:
            desc['mask'] = self.mask
        # The stream field is kept here for compliance. However, since the
        # synchronization is done via calling a cpdef function, which cannot
        # be mock-tested.
        if self.stream is not None:
            if self.stream is cuda.Stream.null:
                desc['stream'] = cuda.runtime.streamLegacy
            elif (not cuda.runtime.is_hip) and self.stream is cuda.Stream.ptds:
                desc['stream'] = cuda.runtime.streamPerThread
            else:
                desc['stream'] = self.stream.ptr
        return desc


@testing.parameterize(
    *testing.product({
        'ndmin': [0, 1, 2, 3],
        'copy': [True, False],
        'xp': [numpy, cupy]
    })
)
class TestArrayPreservationOfShape(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_cupy_array(self, dtype):
        shape = 2, 3
        a = testing.shaped_arange(shape, self.xp, dtype)
        cupy.array(a, copy=self.copy, ndmin=self.ndmin)

        # Check if cupy.ndarray does not alter
        # the shape of the original array.
        assert a.shape == shape


@testing.parameterize(
    *testing.product({
        'ndmin': [0, 1, 2, 3],
        'copy': [True, False],
        'xp': [numpy, cupy]
    })
)
class TestArrayCopy(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_cupy_array(self, dtype):
        a = testing.shaped_arange((2, 3), self.xp, dtype)
        actual = cupy.array(a, copy=self.copy, ndmin=self.ndmin)

        should_copy = (self.xp is numpy) or self.copy
        # TODO(Kenta Oono): Better determination of copy.
        is_copied = not ((actual is a) or (actual.base is a) or
                         (actual.base is a.base and a.base is not None))
        assert should_copy == is_copied


class TestArrayInvalidObject(unittest.TestCase):

    def test_invalid_type(self):
        a = numpy.array([1, 2, 3], dtype=object)
        with self.assertRaises(ValueError):
            cupy.array(a)
