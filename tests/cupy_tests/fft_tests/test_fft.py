import functools
import unittest
import pytest

import numpy as np

import cupy
from cupy import testing
from cupy.fft import config
from cupy.fft.fft import _default_plan_type

import six


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


def _size_last_transform_axis(shape, s, axes):
    if s is not None:
        if s[-1] is not None:
            return s[-1]
    elif axes is not None:
        return shape[axes[-1]]
    return shape[-1]


@testing.parameterize(*testing.product({
    'n': [None, 0, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho', ''],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
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

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(*testing.product({
    'shape': [(10, 10), (10, 5, 10)],
    'data_order': ['F', 'C'],
    'axis': [0, 1, -1],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
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
        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
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

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.gpu
class TestDefaultPlanType(unittest.TestCase):

    @nd_planning_states()
    def test_default_plan_type(self, enable_nd):
        # test cases where nd CUFFT plan is possible
        ca = cupy.ones((16, 16, 16))
        for axes in [(0, 1), (1, 2), None, (0, 1, 2)]:
            plan_type = _default_plan_type(ca, axes=axes)
            if enable_nd:
                self.assertEqual(plan_type, 'nd')
            else:
                self.assertEqual(plan_type, '1d')

        # only a single axis is transformed -> 1d plan preferred
        for axes in [(0, ), (1, ), (2, )]:
            self.assertEqual(_default_plan_type(ca, axes=axes), '1d')

        # non-contiguous axes -> nd plan not possible
        self.assertEqual(_default_plan_type(ca, axes=(0, 2)), '1d')

        # >3 axes transformed -> nd plan not possible
        ca = cupy.ones((2, 4, 6, 8))
        self.assertEqual(_default_plan_type(ca), '1d')

        # first or last axis not included -> nd plan not possible
        self.assertEqual(_default_plan_type(ca, axes=(1, )), '1d')


@testing.gpu
@testing.slow
class TestFftAllocate(unittest.TestCase):

    def test_fft_allocate(self):
        # Check CuFFTError is not raised when the GPU memory is enough.
        # See https://github.com/cupy/cupy/issues/1063
        # TODO(mizuno): Simplify "a" after memory compaction is implemented.
        a = []
        for i in six.moves.range(10):
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
@testing.with_requires('numpy>=1.10.0')
class TestFft2(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft2(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft2(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


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
    {'shape': (2, 3, 4), 's': None, 'axes': (-1, -3), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (0, 1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (4, 3, 2), 'axes': (2, 0, 1), 'norm': 'ortho'},
    {'shape': (2, 3, 4, 5), 's': None, 'axes': None, 'norm': None},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestFftn(unittest.TestCase):

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
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
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestPlanCtxManagerFftn(unittest.TestCase):

    @nd_planning_states()
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes)
            with plan:
                out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn(self, xp, dtype, enable_nd):
        assert config.enable_nd_planning == enable_nd
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            plan = get_fft_plan(a, self.s, self.axes)
            with plan:
                out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)
        else:
            out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @nd_planning_states()
    @testing.for_complex_dtypes()
    def test_fftn_error_on_wrong_plan(self, dtype, enable_nd):
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
        assert 'The CUFFT plan and a.shape do not match' in str(ex.value)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), ],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestPlanCtxManagerFft(unittest.TestCase):

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.fft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft alway returns np.complex128
        if xp == np and dtype is np.complex64:
            out = out.astype(np.complex64)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if xp == cupy:
            from cupyx.scipy.fftpack import get_fft_plan
            shape = (self.n,) if self.n is not None else None
            plan = get_fft_plan(a, shape=shape)
            assert isinstance(plan, cupy.cuda.cufft.Plan1d)
            with plan:
                out = xp.fft.ifft(a, n=self.n, norm=self.norm)
        else:
            out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp == np and dtype is np.complex64:
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

            plan_type = _default_plan_type(a, s=self.s, axes=self.axes)
            if plan_type == 'nd':
                # nd plans have output with contiguity matching the input
                self.assertEqual(out.flags.c_contiguous, a.flags.c_contiguous)
                self.assertEqual(out.flags.f_contiguous, a.flags.f_contiguous)
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

            plan_type = _default_plan_type(a, s=self.s, axes=self.axes)
            if plan_type == 'nd':
                # nd plans have output with contiguity matching the input
                self.assertEqual(out.flags.c_contiguous, a.flags.c_contiguous)
                self.assertEqual(out.flags.f_contiguous, a.flags.f_contiguous)
            else:
                # 1d planning case doesn't guarantee preserved contiguity
                pass


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestRfft(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.n is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


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
@testing.with_requires('numpy>=1.10.0')
class TestRfft2(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfft2(self, xp, dtype):
        # the scaling of old Numpy is incorrect
        if np.__version__ < np.lib.NumpyVersion('1.13.0'):
            if self.s is not None:
                return xp.empty(0)

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft2(a, s=self.s, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)
        return out


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
@testing.with_requires('numpy>=1.10.0')
class TestRfftn(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn(self, xp, dtype):
        if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010 and
                int(cupy.cuda.device.get_compute_capability()) < 70 and
                _size_last_transform_axis(self.shape, self.s, self.axes) == 2):
            pytest.skip('work-around for cuFFT issue')

        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
class TestHfft(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, norm=self.norm)

        if xp == np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {'n': 1, 'd': 1},
    {'n': 10, 'd': 0.5},
    {'n': 100, 'd': 2},
)
@testing.gpu
@testing.with_requires('numpy>=1.10.0')
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
@testing.with_requires('numpy>=1.10.0')
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
