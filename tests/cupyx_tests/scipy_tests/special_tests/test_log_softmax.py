import pytest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


atol = {'default': 1e-6, cupy.float64: 1e-14}
rtol = {'default': 1e-6, cupy.float64: 1e-14}

atol_low = {'default': 1e-6, cupy.float16: 1e-3, cupy.float64: 1e-14}
rtol_low = {'default': 1e-6, cupy.float16: 1e-3, cupy.float64: 1e-14}


@testing.with_requires('scipy')
class TestLogSoftmax:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_ndarray_1(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        a = testing.shaped_random((40, 50), xp, dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_ndarray_2(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        a = xp.array([1000, 1], dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_2d(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        a = testing.shaped_random((5, 3), xp, dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_axis_arg(self, xp, scp, dtype):
        a = xp.array([[100, 1000], [1e-10, 1e-10]])
        return scp.special.log_softmax(a, axis=-1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(
        scipy_name='scp',
        atol=atol_low,
        rtol=rtol_low
    )
    def test_log_softmax_3d(self, xp, scp, dtype):
        if xp.dtype(dtype).kind == 'u':
            # Unsigned integers make underflows in numpy (eventually seen as
            # overflows in `np.exp(tmp)`)
            pytest.skip()
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return scp.special.log_softmax(a, axis=1)
