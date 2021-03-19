import functools

import numpy as np
import pytest

import cupy
from cupy.fft import config
from cupy.fft._fft import (_default_fft_func, _fft, _fftn,
                           _size_last_transform_axis)
from cupy import testing
from cupy.testing._helper import _wraps_partial


@pytest.fixture
def skip_forward_backward(request):
    if request.instance.norm in ('backward', 'forward'):
        if not (np.lib.NumpyVersion(np.__version__) >= '1.20.0'):
            pytest.skip('forward/backward is supported by NumPy 1.20+')


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
        @_wraps_partial(impl, name)
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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(0,), (10, 0), (10,), (10, 10)],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.gpu
class TestFft:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft always returns np.complex128
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
class TestFftOrder:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.fft(a, axis=self.axis)

        # np.fft.fft always returns np.complex128
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
        pytest.skip('avoid CUDA 11 bug')


# Almost identical to the TestFft class, except that
# 1. multi-GPU cuFFT is used
# 2. the tested parameter combinations are adjusted to meet the requirements
@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 0, 64],
    'shape': [(0,), (0, 10), (64,), (4, 64)],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.multi_gpu(2)
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support multi-GPU FFT')
class TestMultiGpuFft:

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft always returns np.complex128
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

        # np.fft.fft always returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out


# Almost identical to the TestFftOrder class, except that
# 1. multi-GPU cuFFT is used
# 2. the tested parameter combinations are adjusted to meet the requirements
@testing.parameterize(*testing.product({
    'shape': [(10, 10), (10, 5, 10)],
    'data_order': ['F', 'C'],
    'axis': [0, 1, -1],
}))
@testing.multi_gpu(2)
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support multi-GPU FFT')
class TestMultiGpuFftOrder:
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

        # np.fft.fft always returns np.complex128
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

        # np.fft.fft always returns np.complex128
        if xp is np and dtype is np.complex64:
            out = out.astype(dtype)

        return out


@testing.gpu
class TestDefaultPlanType:

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


@pytest.mark.skipif(10010 <= cupy.cuda.runtime.runtimeGetVersion() <= 11010,
                    reason='avoid a cuFFT bug (cupy/cupy#3777)')
@testing.gpu
@testing.slow
class TestFftAllocate:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, None), 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': ()},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': ()},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
        {'shape': (0, 5), 's': None, 'axes': None},
        {'shape': (2, 0, 5), 's': None, 'axes': None},
        {'shape': (0, 0, 5), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (0, 5), 'axes': None},
        {'shape': (3, 4), 's': (1, 0), 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestFft2:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, None), 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': [-1, -2]},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': ()},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': ()},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4), 's': (4, 3, 2), 'axes': (2, 0, 1)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
        {'shape': (0, 5), 's': None, 'axes': None},
        {'shape': (2, 0, 5), 's': None, 'axes': None},
        {'shape': (0, 0, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestFftn:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (0, 5), 's': None, 'axes': None},
        {'shape': (2, 0, 5), 's': None, 'axes': None},
        {'shape': (0, 0, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward']})
    )
))
@testing.gpu
class TestPlanCtxManagerFftn:

    @pytest.fixture(autouse=True)
    def skip_buggy(self):
        if cupy.cuda.runtime.is_hip:
            # TODO(leofang): test newer ROCm versions
            if (self.axes == (0, 1) and self.shape == (2, 3, 4)):
                pytest.skip("hipFFT's PlanNd for this case "
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
            pytest.skip('0 in shape')
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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), ],
    'norm': [None, 'backward', 'ortho', 'forward'],
}))
@testing.gpu
class TestPlanCtxManagerFft:

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

        # np.fft.fft always returns np.complex128
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
@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 64],
    'shape': [(64,), (128,)],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.multi_gpu(2)
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support multi-GPU FFT')
class TestMultiGpuPlanCtxManagerFft:

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

        # np.fft.fft always returns np.complex128
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
        if self.norm == '':
            # if norm is invalid, we still get ValueError, but it's raised
            # when checking norm, earlier than the plan check
            return  # skip
        assert 'Target array size does not match the plan.' in str(ex.value)


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': (-3, -2, -1)},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestFftnContiguity:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.gpu
class TestRfft:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,)],
    'norm': [None, 'backward', 'ortho', 'forward'],
}))
@testing.gpu
class TestPlanCtxManagerRfft:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, None), 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestRfft2:

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
            pytest.skip('work-around for cuFFT issue')

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
class TestRfft2EmptyAxes:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, None), 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestRfftn:

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
            pytest.skip('work-around for cuFFT issue')

        a = testing.shaped_random(self.shape, xp, dtype)
        if order == 'F':
            a = xp.asfortranarray(a)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


# Only those tests in which a legit plan can be obtained are kept
@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': (1, None), 'axes': None},
        {'shape': (3, 4), 's': (1, 5), 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2)},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestPlanCtxManagerRfftn:

    @pytest.fixture(autouse=True)
    def skip_buggy(self):
        if cupy.cuda.runtime.is_hip:
            # TODO(leofang): test newer ROCm versions
            if (self.axes == (0, 1) and self.shape == (2, 3, 4)):
                pytest.skip("hipFFT's PlanNd for this case "
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

    @pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                        reason="hipFFT's PlanNd for C2R is buggy")
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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*(
    testing.product_dict([
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (3, 4), 's': None, 'axes': (-2, -1)},
        {'shape': (3, 4), 's': None, 'axes': (-1, -2)},
        {'shape': (3, 4), 's': None, 'axes': (0,)},
        {'shape': (3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None},
        {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None},
        {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1)},
        {'shape': (2, 3, 4), 's': None, 'axes': (-1, -2, -3)},
        {'shape': (2, 3, 4), 's': None, 'axes': (0, 1)},
        {'shape': (2, 3, 4), 's': None, 'axes': None},
        {'shape': (2, 3, 4, 5), 's': None, 'axes': None},
    ],
        testing.product({'norm': [None, 'backward', 'ortho', 'forward', '']})
    )
))
@testing.gpu
class TestRfftnContiguity:

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
class TestRfftnEmptyAxes:

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


@pytest.mark.usefixtures('skip_forward_backward')
@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'backward', 'ortho', 'forward', ''],
}))
@testing.gpu
class TestHfft:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
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
class TestFftfreq:

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
class TestFftshift:

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


class TestThreading:

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
