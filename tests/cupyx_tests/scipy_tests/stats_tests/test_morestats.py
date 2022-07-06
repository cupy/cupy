import cupy

from cupy import testing

import cupyx
import cupyx.scipy.stats  # NOQA

import scipy.stats  # NOQA


atol = {
    'default': 1e-6,
    cupy.float16: 5e-2,
    cupy.float32: 1e-5,
    cupy.float64: 1e-14
}
rtol = {
    'default': 1e-6,
    cupy.float16: 5e-2,
    cupy.float32: 1e-5,
    cupy.float64: 1e-14
}


@testing.with_requires('scipy')
class TestBoxcox_llf:

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-6)
    def test_array_1dim(self, xp, scp, dtype):
        data = testing.shaped_random((2,), xp, dtype=dtype, scale=9)
        lmb = 4.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=5e-3)
    def test_array_1dim_float16(self, xp, scp):
        data = xp.array([2, 3], dtype=xp.float16)
        lmb = 4.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_2dim(self, xp, scp, dtype):
        data = xp.array([[1], [2]], dtype=dtype)
        lmb = 6.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_3dim(self, xp, scp, dtype):
        data = xp.array([
            [
                [1, 2, 3], [4, 5, 6]
            ],
            [
                [7, 8, 9], [10, 11, 12]
            ],
        ], dtype=dtype)
        lmb = 1.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_multi_dim(self, xp, scp, dtype):
        data = xp.array([
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ],
            [
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]]
            ]
        ], dtype=dtype)
        lmb = 9.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_zero_lmb(self, xp, scp, dtype):
        data = xp.array([[1, 2, 3], [6, 8, 9]], dtype=dtype)
        lmb = 0.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_array_empty(self, xp, scp, dtype):
        data = xp.array([], dtype=dtype)
        lmb = 1
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_lmb_neg(self, xp, scp, dtype):
        data = xp.array([198.0, 233.0, 233.0, 392.0])
        lmb = -45
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_lmb_neg2(self, xp, scp, dtype):
        data = xp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        lmb = -3.0
        return scp.stats.boxcox_llf(lmb, data)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_array_empty_neg_lmb(self, xp, scp, dtype):
        data = xp.array([], dtype=dtype)
        lmb = -1.0
        return scp.stats.boxcox_llf(lmb, data)
