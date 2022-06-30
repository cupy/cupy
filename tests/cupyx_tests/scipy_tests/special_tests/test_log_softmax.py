import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA

import scipy.special  # NOQA


atol = {'default': 1e-6, cupy.float64: 1e-14}
rtol = {'default': 1e-6, cupy.float64: 1e-14}


@testing.with_requires('scipy')
class TestLogSoftmax:

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_large_inputs(self, xp, scp, dtype):
        a = xp.arange(4, dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_log_softmax_ndarray(self, xp, scp, dtype):
        a = xp.array([1000, 1], dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_log_softmax_ndarray_2(self, xp, scp, dtype):
        a = xp.array([0, -99], dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_log_softmax_2d(self, xp, scp, dtype):
        a = testing.shaped_random((5, 3), xp, dtype=dtype)
        return scp.special.log_softmax(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-8, rtol=1e-8)
    def test_log_softmax_axis_arg(self, xp, scp, dtype):
        a = xp.array([[100, 1000], [1e-10, 1e-10]])
        return scp.special.log_softmax(a, axis=-1)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    def test_log_softmax_3d(self, xp, scp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return scp.special.log_softmax(a, axis=1)
