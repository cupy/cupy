import numpy
import unittest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    _multiprocess_can_split_ = True

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


@testing.parameterize(
    *testing.product(
        {'src': [float(3.2), int(0), int(4), int(-4), True, False],
         'dst_shape': [(), (0,), (1,), (1, 1), (2, 2)]}))
@testing.gpu
class TestCopytoFromScalar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        xp.copyto(dst, self.src)
        return dst
