import itertools
import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype='?')
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_broadcast(self, xp, dtype):
        a = testing.shaped_arange((3, 1), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_where(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        c = testing.shaped_arange((2, 3, 4), xp, '?')
        xp.copyto(a, b, where=c)
        return a

    def _check_copyto_where_multigpu_raises(self, dtype, ngpus):
        def get_numpy():
            a = testing.shaped_arange((2, 3, 4), numpy, dtype)
            b = testing.shaped_reverse_arange((2, 3, 4), numpy, dtype)
            c = testing.shaped_arange((2, 3, 4), numpy, '?')
            numpy.copyto(a, b, where=c)
            return a

        for dev1, dev2, dev3, dev4 in itertools.product(*[range(ngpus)] * 4):
            if dev1 == dev2 == dev3 == dev4:
                continue
            if not dev1 <= dev2 <= dev3 <= dev4:
                continue

            with cuda.Device(dev1):
                a = testing.shaped_arange((2, 3, 4), cupy, dtype)
            with cuda.Device(dev2):
                b = testing.shaped_reverse_arange((2, 3, 4), cupy, dtype)
            with cuda.Device(dev3):
                c = testing.shaped_arange((2, 3, 4), cupy, '?')
            with cuda.Device(dev4):
                with self.assertRaisesRegex(
                        ValueError,
                        '^Array device must be same as the current device'):
                    cupy.copyto(a, b, where=c)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copyto_where_multigpu_raises(self, dtype):
        self._check_copyto_where_multigpu_raises(dtype, 2)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_multigpu(self, xp, dtype):
        with cuda.Device(0):
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
        with cuda.Device(1):
            b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copyto_multigpu_noncontinguous(self, dtype):
        with cuda.Device(0):
            src = testing.shaped_arange((2, 3, 4), cupy, dtype)
            src = src.swapaxes(0, 1)
        with cuda.Device(1):
            dst = cupy.empty_like(src)
            cupy.copyto(dst, src)

        expected = testing.shaped_arange((2, 3, 4), numpy, dtype)
        expected = expected.swapaxes(0, 1)

        testing.assert_array_equal(expected, src.get())
        testing.assert_array_equal(expected, dst.get())


@testing.parameterize(
    *testing.product(
        {'src': [float(3.2), int(0), int(4), int(-4), True, False, 1 + 1j],
         'dst_shape': [(), (0,), (1,), (1, 1), (2, 2)]}))
@testing.gpu
class TestCopytoFromScalar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        xp.copyto(dst, self.src)
        return dst

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto_where(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        mask = (testing.shaped_arange(
            self.dst_shape, xp, dtype) % 2).astype(xp.bool_)
        xp.copyto(dst, self.src, where=mask)
        return dst
