import math
import sys
import unittest

import cupy
from cupy import testing


@testing.gpu
class TestRanges(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange(self, xp, dtype):
        return xp.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange2(self, xp, dtype):
        return xp.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange3(self, xp, dtype):
        return xp.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange4(self, xp, dtype):
        return xp.arange(20, 2, -3, dtype=dtype)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_int(self, xp):
        return xp.arange(1, 11, 2)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_float(self, xp):
        return xp.arange(1.0, 11.0, 2.0)

    @testing.numpy_cupy_array_equal()
    def test_arange_negative_size(self, xp):
        return xp.arange(3, 1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace2(self, xp, dtype):
        return xp.linspace(10, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_zero_num(self, xp, dtype):
        return xp.linspace(0, 10, 0, dtype=dtype)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_zero_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 0, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_one_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 1, dtype=dtype, endpoint=False,
                              retstep=True)
        self.assertTrue(math.isnan(step))
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_one_num(self, xp, dtype):
        return xp.linspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_no_endpoint(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype, endpoint=False)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 5, dtype=dtype, retstep=True)
        self.assertEqual(step, 2.5)
        return x

    @testing.numpy_cupy_allclose()
    def test_linspace_no_dtype_int(self, xp):
        return xp.linspace(0, 10)

    @testing.numpy_cupy_allclose()
    def test_linspace_no_dtype_float(self, xp):
        return xp.linspace(0.0, 10.0)

    @testing.numpy_cupy_allclose()
    def test_linspace_float_args_with_int_dtype(self, xp):
        return xp.linspace(0.1, 9.1, 11, dtype=int)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_raises()
    def test_linspace_neg_num(self, xp):
        return xp.linspace(0, 10, -1)

    @testing.numpy_cupy_allclose()
    def test_linspace_float_overflow(self, xp):
        return xp.linspace(0., sys.float_info.max / 5, 10, dtype=float)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_array_equal()
    def test_linspace_float_underflow(self, xp):
        # find minimum subnormal number
        x = sys.float_info.min
        while x / 2 > 0:
            x /= 2
        return xp.linspace(0., x, 10, dtype=float)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace(self, xp, dtype):
        return xp.logspace(0, 2, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace2(self, xp, dtype):
        return xp.logspace(2, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_zero_num(self, xp, dtype):
        return xp.logspace(0, 2, 0, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_one_num(self, xp, dtype):
        return xp.logspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_no_endpoint(self, xp, dtype):
        return xp.logspace(0, 2, 5, dtype=dtype, endpoint=False)

    @testing.numpy_cupy_allclose()
    def test_logspace_no_dtype_int(self, xp):
        return xp.logspace(0, 2)

    @testing.numpy_cupy_allclose()
    def test_logspace_no_dtype_float(self, xp):
        return xp.logspace(0.0, 2.0)

    @testing.numpy_cupy_allclose()
    def test_logspace_float_args_with_int_dtype(self, xp):
        return xp.logspace(0.1, 2.1, 11, dtype=int)

    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_raises()
    def test_logspace_neg_num(self, xp):
        return xp.logspace(0, 10, -1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_base(self, xp, dtype):
        return xp.logspace(0, 2, 5, base=2.0, dtype=dtype)


@testing.parameterize(
    *testing.product({
        'indexing': ['xy', 'ij'],
        'copy': [False, True]
    })
)
@testing.gpu
class TestMeshgrid(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_meshgrid0(self, dtype):
        out = cupy.meshgrid(indexing=self.indexing, copy=self.copy)
        assert(out == [])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_meshgrid1(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        return xp.meshgrid(x, indexing=self.indexing, copy=self.copy)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_meshgrid2(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        y = xp.arange(3).astype(dtype)
        return xp.meshgrid(x, y, indexing=self.indexing, copy=self.copy)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_meshgrid3(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        y = xp.arange(3).astype(dtype)
        z = xp.arange(4).astype(dtype)
        return xp.meshgrid(x, y, z, indexing=self.indexing, copy=self.copy)
