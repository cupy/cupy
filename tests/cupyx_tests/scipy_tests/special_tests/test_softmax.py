import pytest

import cupy
from cupy import testing
import cupyx
import cupyx.scipy.special  # NOQA


atol = {
    'default': 1e-6,
    cupy.float16: 1e-3,
}
rtol = {
    'default': 1e-6,
    cupy.float16: 1e-3,
}


@testing.with_requires('scipy')
class TestSoftmax:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_1dim(self, xp, scp, dtype):
        x = xp.arange(400)
        return scp.special.softmax(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_2dim(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        x = testing.shaped_random((5, 4), xp, dtype=dtype, scale=8)
        return scp.special.softmax(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_2dim_float16(self, xp, scp):
        x = testing.shaped_random((5, 4), xp, dtype=xp.float16, scale=8)
        return scp.special.softmax(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_multi_dim(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        x = testing.shaped_random((5, 6, 7, 4), xp, dtype=dtype, scale=8)
        return scp.special.softmax(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-4, rtol=1e-4)
    def test_multi_dim_float16(self, xp, scp):
        x = testing.shaped_random((5, 6, 7, 4), xp, dtype=xp.float16, scale=8)
        return scp.special.softmax(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=atol)
    def test_2dim_with_axis(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        x = testing.shaped_random((5, 4), xp, dtype=dtype, scale=8)
        return scp.special.softmax(x, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_empty(self, xp, scp, dtype):
        x = testing.shaped_random((), xp, dtype=dtype)
        return scp.special.softmax(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zeros_ones(self, xp, scp):
        x = xp.array([0.0, 1.0, -1.0, -0.0])
        return scp.special.softmax(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nans_infs(self, xp, scp):
        x = xp.array([cupy.inf, cupy.nan, -cupy.nan, -cupy.inf])
        return scp.special.softmax(x)
