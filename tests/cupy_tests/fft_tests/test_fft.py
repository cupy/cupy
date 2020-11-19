import functools
import pytest
import string
import unittest

import numpy as np

import cupy
from cupy import testing
from cupy.fft import config
from cupy.fft._fft import (_default_fft_func, _fft, _fftn,
                           _size_last_transform_axis)


def nd_planning_states(states=[True, False], name='enable_nd'):
    """Decorator for parameterized tests with and wihout nd planning

    Tests are repeated with config.enable_nd_planning set to True and False

    Args:
         states(list of bool): The boolean cases to test.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # get original global planning state
            planning_state = config.enable_nd_planning
            try:
                for nd_planning in states:
                    try:
                        # enable or disable nd planning
                        config.enable_nd_planning = nd_planning

                        kw[name] = nd_planning
                        impl(self, *args, **kw)
                    except Exception:
                        print(name, 'is', nd_planning)
                        raise
            finally:
                # restore original global planning state
                config.enable_nd_planning = planning_state

        return test_func
    return decorator


def multi_gpu_config(gpu_configs=None):
    """Decorator for parameterized tests with different GPU configurations.

    Args:
        gpu_configs (list of list): The GPUs to test.

    .. notes:
        The decorated tests are skipped if no or only one GPU is available.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            use_multi_gpus = config.use_multi_gpus
            _devices = config._devices

            try:
                for gpus in gpu_configs:
                    try:
                        nGPUs = len(gpus)
                        assert nGPUs >= 2, 'Must use at least two gpus'
                        config.use_multi_gpus = True
                        config.set_cufft_gpus(gpus)
                        self.gpus = gpus

                        impl(self, *args, **kw)
                    except Exception:
                        print('GPU config is:', gpus)
                        raise
            finally:
                config.use_multi_gpus = use_multi_gpus
                config._devices = _devices
                del self.gpus

        return test_func
    return decorator


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(0,), (10, 0), (10,), (10, 10)],
    'norm': [None, 'ortho', ''],
}))
@testing.gpu
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(*testing.product({
    'shape': [(0, 10), (10, 0, 10), (10, 10), (10, 5, 10)],
    'data_order': ['F', 'C'],
    'axis': [0, 1, -1],
}))
@testing.gpu
class TestFftOrder(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.fft(a, axis=self.axis)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.ifft(a, axis=self.axis)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


# See #3757 and NVIDIA internal ticket 3093094
def _skip_multi_gpu_bug(shape, gpus):
    # avoid CUDA 11.0 (will be fixed by CUDA 11.2) bug triggered by
    # - batch = 1
    # - gpus = [1, 0]
    if (11000 <= cupy.cuda.runtime.runtimeGetVersion() < 11200
            and len(shape) == 1
            and gpus == [1, 0]):
        raise unittest.SkipTest('avoid CUDA 11 bug')


# Almost identical to the TestFft class, except that
# 1. multi-GPU cuFFT is used
# 2. the tested parameter combinations are adjusted to meet the requirements
@unittest.skipIf(cupy.cuda.runtime.is_hip,
                 'hipFFT does not support multi-GPU FFT')
@testing.parameterize(*testing.product({
    'n': [None, 0, 64],
    'shape': [(0,), (0, 10), (64,), (4, 64)],
    'norm': [None, 'ortho', ''],
}))
@testing.multi_gpu(2)
class TestMultiGpuFft(unittest.TestCase):

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires('numpy!=1.17.0')
    @testing.with_requires('numpy!=1.17.1')
    def test_ifft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out


# Almost identical to the TestFftOrder class, except that
# 1. multi-GPU cuFFT is used
# 2. the tested parameter combinations are adjusted to meet the requirements
@unittest.skipIf(cupy.cuda.runtime.is_hip,
                 'hipFFT does not support multi-GPU FFT')
@testing.parameterize(*testing.product({
    'shape': [(10, 10), (10, 5, 10)],
    'data_order': ['F', 'C'],
    'axis': [0, 1, -1],
}))
@testing.multi_gpu(2)
class TestMultiGpuFftOrder(unittest.TestCase):
    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.fft(a, axis=self.axis)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.ifft(a, axis=self.axis)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out


@testing.gpu
class TestDefaultPlanType(unittest.TestCase):

    @nd_planning_states()
    def test_default_fft_func(self, enable_nd):
        # test cases where nd cuFFT plan is possible
        ca = cupy.ones((16, 16, 16))
        for axes in [(0, 1), (1, 2), None, (0, 1, 2)]:
            fft_func = _default_fft_func(ca, axes=axes)
            if enable_nd:
                # TODO(leofang): test newer ROCm versions
                if axes == (0, 1) and cupy.cuda.runtime.is_hip:
                    assert fft_func is _fft
                else:
                    assert fft_func is _fftn
            else:
                assert fft_func is _fft

        # only a single axis is transformed -> 1d plan preferred
        for axes in [(0, ), (1, ), (2, )]:
            assert _default_fft_func(ca, axes=axes) is _fft

        # non-contiguous axes -> nd plan not possible
        assert _default_fft_func(ca, axes=(0, 2)) is _fft

        # >3 axes transformed -> nd plan not possible
        ca = cupy.ones((2, 4, 6, 8))
        assert _default_fft_func(ca) is _fft

        # first or last axis not included -> nd plan not possible
        assert _default_fft_func(ca, axes=(1, )) is _fft

        # for rfftn
        ca = cupy.random.random((4, 2, 6))
        for s, axes in zip([(3, 4), None, (8, 7, 5)],
                           [(-2, -1), (0, 1), None]):
            fft_func = _default_fft_func(ca, s=s, axes=axes, value_type='R2C')
            if enable_nd:
                # TODO(leofang): test newer ROCm versions
                if axes == (0, 1) and cupy.cuda.runtime.is_hip:
                    assert fft_func is _fft
                else:
                    assert fft_func is _fftn
            else:
                assert fft_func is _fft

        # nd plan not possible if last axis is not 0 or ndim-1
        assert _default_fft_func(ca, axes=(2, 1), value_type='R2C') is _fft

        # for irfftn
        ca = cupy.random.random((4, 2, 6)).astype(cupy.complex128)
        for s, axes in zip([(3, 4), None, (8, 7, 5)],
                           [(-2, -1), (0, 1), None]):
            fft_func = _default_fft_func(ca, s=s, axes=axes, value_type='C2R')
            if enable_nd:
                # To get around hipFFT's bug, we don't use PlanNd for C2R
                # TODO(leofang): test newer ROCm versions
                if cupy.cuda.runtime.is_hip:
                    assert fft_func is _fft
                else:
                    assert fft_func is _fftn
            else:
                assert fft_func is _fft

        # nd plan not possible if last axis is not 0 or ndim-1
        assert _default_fft_func(ca, axes=(2, 1), value_type='C2R') is _fft


@testing.gpu
@testing.slow
class TestFftAllocate(unittest.TestCase):

    def test_fft_allocate(self):
        # Check CuFFTError is not raised when the GPU memory is enough.
        # See https://github.com/cupy/cupy/issues/1063
        # TODO(mizuno): Simplify "a" after memory compaction is implemented.
        a = []
        for i in range(10):
            a.append(cupy.empty(100000000))
        del a
        b = cupy.empty(100000007, dtype=cupy.float32)
        cupy.fft.fft(b)
        # Free huge memory for slow test
        del b
        cupy.get_default_memory_pool().free_all_blocks()


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (3, 4), 's': None, 'axes': (), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (), 'norm': None},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (0, 0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (0, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 0), 'axes': None, 'norm': None},
)
@testing.gpu
class TestFft2(unittest.TestCase):

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.fft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.ifft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': [-1, -2], 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': (), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (4, 3, 2), 'axes': (2, 0, 1), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (0, 0, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
class TestFftn(unittest.TestCase):

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 0, 5), 's': None, 'axes': None, 'norm': None},
    {'shape': (0, 0, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
class TestPlanCtxManagerFftn(unittest.TestCase):

    def setUp(self):
        if cupy.cuda.runtime.is_hip:
            # TODO(leofang): test newer ROCm versions
            if (self.axes == (0, 1) and self.shape == (2, 3, 4)):
                raise unittest.SkipTest("hipFFT's PlanNd for this case "
                                        "is buggy, so Plan1d is generated "
                                        "instead")

    @nd_planning_states()
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes)
            with plan:
                out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes)
            with plan:
                out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_complex_dtypes()
    def test_fftn_error_on_wrong_plan(self, dtype, enable_nd):
        if 0 in self.shape:
            raise unittest.SkipTest('0 in shape')
        # This test ensures the context manager plan is picked up

        from cupyx.scipy.fftpack import get_fft_plan
        from cupy.fft import fftn
        assert config.enable_nd_planning == enable_nd

        # can't get a plan, so skip
        if self.axes is not None:
            if self.s is not None:
                if len(self.s) != len(self.axes):
                    return
            elif len(self.shape) != len(self.axes):
                return

        a = testing.shaped_random(self.shape, cupy, dtype)
        bad_in_shape = tuple(2*i for i in self.shape)
        if self.s is None:
            bad_out_shape = bad_in_shape
        else:
            bad_out_shape = tuple(2*i for i in self.s)
        b = testing.shaped_random(bad_in_shape, cupy, dtype)
        plan_wrong = get_fft_plan(b, bad_out_shape, self.axes)

        with pytest.raises(ValueError) as ex, plan_wrong:
            fftn(a, s=self.s, axes=self.axes, norm=self.norm)
        # targeting a particular error
        assert 'The cuFFT plan and a.shape do not match' in str(ex.value)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), ],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestPlanCtxManagerFft(unittest.TestCase):

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.fft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.ifft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @testing.for_complex_dtypes()
    def test_fft_error_on_wrong_plan(self, dtype):
        # This test ensures the context manager plan is picked up

        from cupyx.scipy.fftpack import get_fft_plan
        from cupy.fft import fft

        a = testing.shaped_random(self.shape, cupy, dtype)
        bad_shape = tuple(5*i for i in self.shape)
        b = testing.shaped_random(bad_shape, cupy, dtype)
        plan_wrong = get_fft_plan(b)
        assert isinstance(plan_wrong, cupy.cuda.cufft.Plan1d)

        with pytest.raises(ValueError) as ex, plan_wrong:
            fft(a, n=self.n, norm=self.norm)
        # targeting a particular error
        assert 'Target array size does not match the plan.' in str(ex.value)


# Almost identical to the TestPlanCtxManagerFft class, except that
# 1. multi-GPU cuFFT is used
# 2. the tested parameter combinations are adjusted to meet the requirements
@unittest.skipIf(cupy.cuda.runtime.is_hip,
                 'hipFFT does not support multi-GPU FFT')
@testing.parameterize(*testing.product({
    'n': [None, 64],
    'shape': [(64,), (128,)],
    'norm': [None, 'ortho'],
}))
@testing.multi_gpu(2)
class TestMultiGpuPlanCtxManagerFft(unittest.TestCase):
    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.fft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.ifft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp is np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    def test_fft_error_on_wrong_plan(self, dtype):
        # This test ensures the context manager plan is picked up

        from cupyx.scipy.fftpack import get_fft_plan
        from cupy.fft import fft

        a = testing.shaped_random(self.shape, cupy, dtype)
        bad_shape = tuple(4*i for i in self.shape)
        b = testing.shaped_random(bad_shape, cupy, dtype)
        plan_wrong = get_fft_plan(b)
        assert isinstance(plan_wrong, cupy.cuda.cufft.Plan1d)

        with pytest.raises(ValueError) as ex, plan_wrong:
            fft(a, n=self.n, norm=self.norm)
        # targeting a particular error
        assert 'Target array size does not match the plan.' in str(ex.value)


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': (-3, -2, -1), 'norm': None},
)
@testing.gpu
class TestFftnContiguity(unittest.TestCase):

    @nd_planning_states([True])
    @testing.for_all_dtypes()
    def test_fftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:
            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.fftn(a, s=self.s, axes=self.axes)

            fft_func = _default_fft_func(a, s=self.s, axes=self.axes)
            if fft_func is _fftn:
                # nd plans have output with contiguity matching the input
                assert out.flags.c_contiguous == a.flags.c_contiguous
                assert out.flags.f_contiguous == a.flags.f_contiguous
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass

    @nd_planning_states([True])
    @testing.for_all_dtypes()
    def test_ifftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:

            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.ifftn(a, s=self.s, axes=self.axes)

            fft_func = _default_fft_func(a, s=self.s, axes=self.axes)
            if fft_func is _fftn:
                # nd plans have output with contiguity matching the input
                assert out.flags.c_contiguous == a.flags.c_contiguous
                assert out.flags.f_contiguous == a.flags.f_contiguous
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestPlanCtxManagerRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape, value_type='R2C')
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.rfft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)

        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape, value_type='C2R')
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.irfft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    def test_rfft_error_on_wrong_plan(self, dtype):
        # This test ensures the context manager plan is picked up

        from cupyx.scipy.fftpack import get_fft_plan
        from cupy.fft import rfft

        a = testing.shaped_random(self.shape, cupy, dtype)
        bad_shape = tuple(5*i for i in self.shape)
        b = testing.shaped_random(bad_shape, cupy, dtype)
        plan_wrong = get_fft_plan(b, value_type='R2C')
        assert isinstance(plan_wrong, cupy.cuda.cufft.Plan1d)

        with pytest.raises(ValueError) as ex, plan_wrong:
            rfft(a, n=self.n, norm=self.norm)
        # targeting a particular error
        assert 'Target array size does not match the plan.' in str(ex.value)


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
class TestRfft2(unittest.TestCase):

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.rfft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010
                and int(cupy.cuda.device.get_compute_capability()) < 70
                and _size_last_transform_axis(
                    self.shape, self.s, self.axes) == 2):
            raise unittest.SkipTest('work-around for cuFFT issue')

        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.irfft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)
        return out


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': (), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (), 'norm': None},
)
@testing.gpu
class TestRfft2EmptyAxes(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    def test_rfft2(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.rfft2(a, s=self.s, axes=self.axes, norm=self.norm)

    @testing.for_all_dtypes()
    def test_irfft2(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.irfft2(a, s=self.s, axes=self.axes, norm=self.norm)


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
class TestRfftn(unittest.TestCase):

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype, order, enable_nd):
        assert config.enable_nd_planning == enable_nd
        if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010
                and int(cupy.cuda.device.get_compute_capability()) < 70
                and _size_last_transform_axis(
                    self.shape, self.s, self.axes) == 2):
            raise unittest.SkipTest('work-around for cuFFT issue')

        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


# Only those tests in which a legit plan can be obtained are kept
@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
)
@testing.gpu
class TestPlanCtxManagerRfftn(unittest.TestCase):

    def setUp(self):
        if cupy.cuda.runtime.is_hip:
            # TODO(leofang): test newer ROCm versions
            if (self.axes == (0, 1) and self.shape == (2, 3, 4)):
                raise unittest.SkipTest("hipFFT's PlanNd for this case "
                                        "is buggy, so Plan1d is generated "
                                        "instead")

    @nd_planning_states()
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes, value_type='R2C')
            with plan:
                out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @unittest.skipIf(cupy.cuda.runtime.is_hip,
                     "hipFFT's PlanNd for C2R is buggy")
    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes, value_type='C2R')
            with plan:
                out = xp.fft.irfftn(
                    a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    # TODO(leofang): write test_rfftn_error_on_wrong_plan()?


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-1, -2), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (0,), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
class TestRfftnContiguity(unittest.TestCase):

    @nd_planning_states([True])
    @testing.for_float_dtypes()
    def test_rfftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:
            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.rfftn(a, s=self.s, axes=self.axes)

            fft_func = _default_fft_func(a, s=self.s, axes=self.axes,
                                         value_type='R2C')
            if fft_func is _fftn:
                # nd plans have output with contiguity matching the input
                assert out.flags.c_contiguous == a.flags.c_contiguous
                assert out.flags.f_contiguous == a.flags.f_contiguous
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass

    @nd_planning_states([True])
    @testing.for_all_dtypes()
    def test_ifftn_orders(self, dtype, enable_nd):
        for order in ['C', 'F']:

            a = testing.shaped_random(self.shape, cupy, dtype)
            if order == 'F':
                a = cupy.asfortranarray(a)
            out = cupy.fft.irfftn(a, s=self.s, axes=self.axes)

            fft_func = _default_fft_func(a, s=self.s, axes=self.axes,
                                         value_type='C2R')
            if fft_func is _fftn:
                # nd plans have output with contiguity matching the input
                assert out.flags.c_contiguous == a.flags.c_contiguous
                assert out.flags.f_contiguous == a.flags.f_contiguous
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': (), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (), 'norm': None},
)
@testing.gpu
class TestRfftnEmptyAxes(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    def test_rfftn(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

    @testing.for_all_dtypes()
    def test_irfftn(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
class TestHfft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'n': 1, 'd': 1},
    {'n': 10, 'd': 0.5},
    {'n': 100, 'd': 2},
)
@testing.gpu
class TestFftfreq(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftfreq(self, xp, dtype):
        out = xp.fft.fftfreq(self.n, self.d)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfftfreq(self, xp, dtype):
        out = xp.fft.rfftfreq(self.n, self.d)

        return out


@testing.parameterize(
    {'shape': (5,), 'axes': None},
    {'shape': (5,), 'axes': 0},
    {'shape': (10,), 'axes': None},
    {'shape': (10,), 'axes': 0},
    {'shape': (10, 10), 'axes': None},
    {'shape': (10, 10), 'axes': 0},
    {'shape': (10, 10), 'axes': (0, 1)},
)
@testing.gpu
class TestFftshift(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftshift(x, self.axes)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftshift(x, self.axes)

        return out


class TestThreading(unittest.TestCase):

    def test_threading1(self):
        import threading
        from cupy.cuda.cufft import get_current_plan

        def thread_get_curr_plan():
            return get_current_plan()

        new_thread = threading.Thread(target=thread_get_curr_plan)
        new_thread.start()

    def test_threading2(self):
        import threading

        a = cupy.arange(100, dtype=cupy.complex64).reshape(10, 10)

        def thread_do_fft():
            b = cupy.fft.fftn(a)
            return b

        new_thread = threading.Thread(target=thread_do_fft)
        new_thread.start()


_load_callback = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= 2.5;
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_load_callback_with_aux = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= *((${aux_type}*)callerInfo);
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_load_callback_with_aux2 = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= ((${aux_type}*)callerInfo)[offset];
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_store_callback = r'''
__device__ void CB_ConvertOutput(
    void *dataOut, size_t offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= 3.8;
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = CB_ConvertOutput;
'''

_store_callback_with_aux = r'''
__device__ void CB_ConvertOutput(
    void *dataOut, size_t offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= *((${aux_type}*)callerInfo);
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = CB_ConvertOutput;
'''


def _set_load_cb(code, element, data_type, callback_type, aux_type=None):
    return string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        load_type=callback_type,
        element=element)


def _set_store_cb(code, element, data_type, callback_type, aux_type=None):
    return string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        store_type=callback_type,
        element=element)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10, 7), (10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.with_requires('cython>=0.29.0')
@testing.gpu
@unittest.skipIf(cupy.cuda.runtime.is_hip,
                 'hipFFT does not support callbacks')
class Test1dCallbacks(unittest.TestCase):

    def _test_load_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR')
        else:
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
        cb_load = _set_load_cb(code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                if fft_func != 'irfft':
                    out = out.astype(np.complex64)
                else:
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(cb_load=cb_load):
                out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfft')

    def _test_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        code = _store_callback
        if dtype == np.complex64:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                types = ('x', 'cufftReal', 'cufftCallbackStoreR')
        elif dtype == np.complex128:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
            else:
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        cb_store = _set_store_cb(code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(cb_store=cb_store):
                out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfft')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback
        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_store=cb_store):
                out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_aux(self, xp, dtype):
        fft = xp.fft.fft
        c = _load_callback_with_aux2
        if dtype == np.complex64:
            cb_load = _set_load_cb(
                c, 'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
        else:
            cb_load = _set_load_cb(
                c, 'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ', 'double')

        a = testing.shaped_random(self.shape, xp, dtype)
        out_last = self.n if self.n is not None else self.shape[-1]
        out_shape = list(self.shape)
        out_shape[-1] = out_last
        last_min = min(self.shape[-1], out_last)
        b = xp.arange(np.prod(out_shape), dtype=xp.dtype(dtype).char.lower())
        b = b.reshape(out_shape)
        if xp is np:
            x = np.zeros(out_shape, dtype=dtype)
            x[..., 0:last_min] = a[..., 0:last_min]
            x.real *= b
            out = fft(x, n=self.n, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                out = out.astype(np.complex64)
        else:
            with xp.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_load_aux_arr=b):
                out = fft(a, n=self.n, norm=self.norm)

        return out

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = (
                    'x', 'cufftReal', 'cufftCallbackLoadR', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackLoadD', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        else:  # irfft
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x', 'cufftReal', 'cufftCallbackStoreR', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x', 'cufftDoubleReal',
                               'cufftCallbackStoreD', 'double')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_store=cb_store,
                    cb_load_aux_arr=load_aux, cb_store_aux_arr=store_aux):
                out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfft')


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
)
@testing.with_requires('cython>=0.29.0')
@testing.gpu
@unittest.skipIf(cupy.cuda.runtime.is_hip,
                 'hipFFT does not support callbacks')
class TestNdCallbacks(unittest.TestCase):

    def _test_load_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR')
        else:
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
        cb_load = _set_load_cb(load_code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                if fft_func != 'irfftn':
                    out = out.astype(np.complex64)
                else:
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(cb_load=cb_load):
                out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfftn')

    def _test_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        store_code = _store_callback
        if dtype == np.complex64:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                types = ('x', 'cufftReal', 'cufftCallbackStoreR')
        elif dtype == np.complex128:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
            else:
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        cb_store = _set_store_cb(store_code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(cb_store=cb_store):
                out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback
        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_store=cb_store):
                out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = (
                    'x', 'cufftReal', 'cufftCallbackLoadR', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackLoadD', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        else:  # irfftn
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x', 'cufftReal', 'cufftCallbackStoreR', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x', 'cufftDoubleReal',
                               'cufftCallbackStoreD', 'double')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with xp.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_store=cb_store,
                    cb_load_aux_arr=load_aux, cb_store_aux_arr=store_aux):
                out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfftn')
