import unittest
import pytest

from cupy import testing
import cupyx.scipy.fftpack  # NOQA
from cupy.fft.fft import _default_plan_type

if cupyx.scipy._scipy_available:
    import scipy.fftpack  # NOQA


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
}))
@testing.gpu
@testing.with_requires('scipy>=0.19.0')
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.fft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.fft(x, n=self.n, axis=self.axis,
                               overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if scp is cupyx.scipy:
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            out = scp.fftpack.fft(x, n=self.n, axis=self.axis, plan=plan)
        else:  # scipy
            out = scp.fftpack.fft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        if scp is cupyx.scipy:
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            x = scp.fftpack.fft(x, n=self.n, axis=self.axis,
                                overwrite_x=True, plan=plan)
        else:  # scipy
            x = scp.fftpack.fft(x, n=self.n, axis=self.axis,
                                overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.fft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.fft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.ifft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.ifft(x, n=self.n, axis=self.axis,
                                overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if scp is cupyx.scipy:
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            out = scp.fftpack.ifft(x, n=self.n, axis=self.axis, plan=plan)
        else:  # scipy
            out = scp.fftpack.ifft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        if scp is cupyx.scipy:
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            x = scp.fftpack.ifft(x, n=self.n, axis=self.axis,
                                 overwrite_x=True, plan=plan)
        else:  # scipy
            x = scp.fftpack.ifft(x, n=self.n, axis=self.axis,
                                 overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return x
        x_orig = x.copy()
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.ifft(x, n=self.n, axis=self.axis)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.ifft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_complex_dtypes()
    def test_fft_multiple_plan_error(self, dtype):
        # hack: avoid testing the cases when the output array is of size 0
        # because cuFFT and numpy raise different kinds of exceptions
        if self.n == 0:
            return
        import cupy
        import cupyx.scipy.fftpack as fftpack
        x = testing.shaped_random(self.shape, cupy, dtype)
        plan = fftpack.get_fft_plan(x, shape=self.n, axes=self.axis)
        with pytest.raises(RuntimeError) as ex, plan:
            fftpack.fft(x, n=self.n, axis=self.axis, plan=plan)
        assert 'Use the cuFFT plan either as' in str(ex.value)


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
    {'shape': (3, 4), 's': None, 'axes': (0,)},
    {'shape': (2, 3, 4), 's': None, 'axes': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
)
@testing.gpu
@testing.with_requires('scipy>=0.19.0')
class TestFft2(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft2(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.fft2(x, shape=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft2_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.fft2(x, shape=self.s, axes=self.axes,
                                overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft2_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            out = scp.fftpack.fft2(x, shape=self.s, axes=self.axes, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            out = scp.fftpack.fft2(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft2_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            x = scp.fftpack.fft2(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            x = scp.fftpack.fft2(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fft2_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.fft2(x, shape=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.fft2(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft2(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft2_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.ifft2(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft2_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            out = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            out = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft2_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            x = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes,
                                  overwrite_x=True, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            x = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes,
                                  overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifft2_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.ifft2(x, shape=self.s, axes=self.axes)
        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
    {'shape': (3, 4), 's': None, 'axes': (0,)},
    {'shape': (2, 3, 4), 's': None, 'axes': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
)
@testing.gpu
@testing.with_requires('scipy>=0.19.0')
class TestFftn(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fftn(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.fftn(x, shape=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fftn_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.fftn(x, shape=self.s, axes=self.axes,
                                overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fftn_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            out = scp.fftpack.fftn(x, shape=self.s, axes=self.axes, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            out = scp.fftpack.fftn(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fftn_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            x = scp.fftpack.fftn(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            x = scp.fftpack.fftn(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_fftn_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.fftn(x, shape=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.fftn(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifftn(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifftn_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.ifftn(x, shape=self.s, axes=self.axes,
                                 overwrite_x=True)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifftn_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            out = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            out = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifftn_overwrite_plan(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            import cupy.fft.config as config
            config.enable_nd_planning = False  # use explicit plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            x = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes,
                                  overwrite_x=True, plan=plan)
            config.enable_nd_planning = True  # default
        else:  # scipy
            x = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes,
                                  overwrite_x=True)
        return x

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_ifftn_plan_manager(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return x
        if scp is cupyx.scipy:
            from cupy.cuda.cufft import get_current_plan
            plan = scp.fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
            with plan:
                assert id(plan) == id(get_current_plan())
                out = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes)
            assert get_current_plan() is None
        else:  # scipy
            out = scp.fftpack.ifftn(x, shape=self.s, axes=self.axes)
        return out

    @testing.for_complex_dtypes()
    def test_fftn_multiple_plan_error(self, dtype):
        import cupy
        import cupyx.scipy.fftpack as fftpack
        x = testing.shaped_random(self.shape, cupy, dtype)
        # hack: avoid testing the cases when getting a cuFFT plan is impossible
        if _default_plan_type(x, s=self.s, axes=self.axes) != 'nd':
            return
        plan = fftpack.get_fft_plan(x, shape=self.s, axes=self.axes)
        with pytest.raises(RuntimeError) as ex, plan:
            fftpack.fftn(x, shape=self.s, axes=self.axes, plan=plan)
        assert 'Use the cuFFT plan either as' in str(ex.value)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(9,), (10,), (10, 9), (10, 10)],
    'axis': [-1, 0],
}))
@testing.gpu
@testing.with_requires('scipy>=0.19.0')
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_rfft(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.rfft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-6, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_rfft_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.rfft(x, n=self.n, axis=self.axis,
                                overwrite_x=True)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_irfft(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = scp.fftpack.irfft(x, n=self.n, axis=self.axis)
        testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False, scipy_name='scp')
    def test_irfft_overwrite(self, xp, scp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        return scp.fftpack.irfft(x, n=self.n, axis=self.axis,
                                 overwrite_x=True)
