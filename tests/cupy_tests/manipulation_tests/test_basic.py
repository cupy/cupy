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
         'dst_shape': [(), (0,), (1,), (1, 1), (2, 2)],
         'dst_dtype': testing.helper._make_all_dtypes(False, False)}))
@testing.gpu
class TestCopytoFromScalar(unittest.TestCase):

    def _do_copyto(self, xp):
        dst = xp.ones(self.dst_shape, dtype=self.dst_dtype)
        xp.copyto(dst, self.src)
        return dst

    def test_copyto(self):
        try:
            dst_expected = self._do_copyto(numpy)
            success_expected = True
        except TypeError:
            dst_expected = None
            success_expected = False

        if success_expected:
            # Numpy succeeds; expected to succeed
            testing.array.assert_array_equal(
                dst_expected,
                self._do_copyto(cupy))
        else:
            # Numpy fails; expected to fail
            with self.assertRaises(TypeError):
                self._do_copyto(cupy)
