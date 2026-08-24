from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cuda_compute_common
from cupy._core import _cuda_compute_reduction


# This test class and its children below only test if the cuda.compute
# backend can be used or not; they don't verify its correctness as it's
# already extensively covered by existing tests (run with
# CUPY_ACCELERATORS=cuda_compute).
class CudaComputeReductionTestBase:

    @pytest.fixture(autouse=True)
    def configure(self):
        if _cuda_compute_common._get_cuda_compute() is None:
            pytest.skip('cuda.compute (cuda-cccl) not found')

        self.can_use = (
            _cuda_compute_reduction._can_use_cuda_compute_reduction)

        self.old_routine_accelerators = (
            _accelerator.get_routine_accelerators())
        self.old_accelerators = _accelerator.get_reduction_accelerators()
        # routine-level accelerators (cub, cutensor) intercept sum before
        # the reduction machinery; disable them so the reduction-level
        # cuda.compute accelerator is reachable
        _accelerator.set_routine_accelerators([])
        _accelerator.set_reduction_accelerators(['cuda_compute'])
        yield
        _accelerator.set_routine_accelerators(self.old_routine_accelerators)
        _accelerator.set_reduction_accelerators(self.old_accelerators)


class TestCudaComputeReductionCanUse(CudaComputeReductionTestBase):

    def _test_can_use(self, in_shape, reduce_axis, out_axis, expected,
                      order='C'):
        in_args = [cupy.empty(in_shape, dtype='f', order=order)]
        out_args = [cupy.empty((), dtype='f')]
        assert self.can_use(
            in_args, out_args, reduce_axis, out_axis) is expected

    def test_can_use_full_reduction(self):
        self._test_can_use((100,), (0,), (), True)
        self._test_can_use((10, 20), (0, 1), (), True)
        self._test_can_use((10, 20), (0, 1), (), True, order='F')

    def test_can_use_segmented(self):
        # trailing reduce axes on C-contiguous input: served
        self._test_can_use((10, 20), (1,), (0,), True)
        self._test_can_use((4, 5, 6), (1, 2), (0,), True)

    def test_can_use_two_inputs(self):
        a = cupy.empty((100,), dtype='f')
        b = cupy.empty((100,), dtype='f')
        c = cupy.empty((), dtype='f')
        assert self.can_use([a, b], [c], (0,), ()) is False


_scalar_acc_cases = [
    ('sum', {}),
    ('sum', {'dtype': 'q'}),
    ('prod', {}),
    ('nansum', {}),
    ('nanprod', {}),
    ('all', {}),
    ('any', {}),
    ('count_nonzero', {}),
    ('min', {}),
    ('max', {}),
    ('argmin', {}),
    ('argmax', {}),
]


class TestCudaComputeReductionRoutines(CudaComputeReductionTestBase):

    def _dispatch_and_compare(self, routine, kwargs, a_np, times_called=1,
                              order='C'):
        a = cupy.asarray(a_np, order=order)
        func_name = ('cupy._core._cuda_compute_reduction'
                     '._cuda_compute_reduce')

        orig = _cuda_compute_reduction._cuda_compute_reduce

        def must_succeed(*args, **kw):
            # a False/None return means the backend silently declined
            # and the generic kernel computed the result instead
            ret = orig(*args, **kw)
            assert ret, 'cuda.compute declined a supported reduction'
            return ret

        with testing.AssertFunctionIsCalled(
                func_name, wraps=must_succeed, times_called=times_called):
            result = getattr(cupy, routine)(a, **kwargs)
        expected = getattr(numpy, routine)(a_np, **kwargs)
        testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(('routine', 'kwargs'), _scalar_acc_cases)
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_scalar_accumulator_routines(self, routine, kwargs):
        a = numpy.arange(1, 33, dtype='f' if not kwargs else 'i')
        self._dispatch_and_compare(routine, kwargs, a)

    @pytest.mark.parametrize('dtype', ['e', 'f', 'd', 'F', 'D'])
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_sum_dtypes(self, dtype):
        a = testing.shaped_random((1000,), numpy, dtype=dtype, seed=0)
        self._dispatch_and_compare('sum', {}, a)

    @pytest.mark.parametrize('dtype', ['f', 'd', 'F'])
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_sum_to_complex_accumulator(self, dtype):
        a = testing.shaped_random((1000,), numpy, dtype=dtype, seed=0)
        self._dispatch_and_compare('sum', {'dtype': 'D'}, a)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_sum_complex_to_real_declines(self):
        from cupy.cuda.compiler import CompileException
        a = cupy.ones((100,), dtype='F')
        orig = _cuda_compute_reduction._cuda_compute_reduce
        seen = {}

        def record(*args, **kw):
            seen['ret'] = orig(*args, **kw)
            return seen['ret']

        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=record, times_called=1):
            with pytest.raises(CompileException):
                cupy.sum(a, dtype='q')
        assert seen['ret'] is False

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_norm_uses_post_op(self):
        a_np = testing.shaped_random((1000,), numpy, dtype='d', seed=0)
        a = cupy.asarray(a_np)
        orig = _cuda_compute_reduction._cuda_compute_reduce

        def must_succeed(*args, **kw):
            ret = orig(*args, **kw)
            assert ret, 'cuda.compute declined a supported reduction'
            return ret

        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=must_succeed, times_called=1):
            result = cupy.linalg.norm(a)
        testing.assert_allclose(result, numpy.linalg.norm(a_np), rtol=1e-6)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_strided_out_falls_back(self):
        a_np = testing.shaped_random((3, 4), numpy, dtype='f', seed=0)
        a = cupy.asarray(a_np)
        big = cupy.zeros((4, 8), dtype='f')
        strided_out = big[0, ::2]
        func = _cuda_compute_reduction._cuda_compute_reduce
        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=func, times_called=0):  # declined by can_use
            cupy.sum(a, axis=0, out=strided_out)
        testing.assert_allclose(
            strided_out, numpy.sum(a_np, axis=0), rtol=1e-6)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_accelerator_order_respected(self):
        # with cuda_compute listed before cub in both lists (as the
        # env var produces), cuda.compute must get first refusal
        _accelerator.set_routine_accelerators(['cuda_compute', 'cub'])
        _accelerator.set_reduction_accelerators(['cuda_compute', 'cub'])
        a = cupy.ones((1000,), dtype='f')
        func = _cuda_compute_reduction._cuda_compute_reduce
        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=func, times_called=1):
            result = a.sum()
        assert result == 1000.0

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_full_reduction_f_order(self):
        a = testing.shaped_random((30, 40), numpy, dtype='d', seed=0)
        self._dispatch_and_compare('sum', {}, a, order='F')

    @pytest.mark.parametrize('dtype', ['e', 'f', 'd'])
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_nanmean(self, dtype):
        # nanmean_st carries the non-NaN count in the accumulator, so
        # unlike mean no host-side divisor is needed
        a = testing.shaped_random((1000,), numpy, dtype=dtype, seed=0)
        a[::7] = numpy.nan
        rtol = 1e-2 if dtype == 'e' else 1e-6
        acc = cupy.asarray(a)
        func = _cuda_compute_reduction._cuda_compute_reduce

        def must_succeed(*args, **kw):
            ret = func(*args, **kw)
            assert ret, 'cuda.compute declined a supported reduction'
            return ret

        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=must_succeed, times_called=1):
            result = cupy.nanmean(acc)
        testing.assert_allclose(result, numpy.nanmean(a), rtol=rtol)

    @pytest.mark.parametrize('routine', ['nanmin', 'nanmax'])
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_nan_min_max(self, routine):
        # two dispatches: the reduction itself, then the wrapper's
        # isnan(res).any() all-NaN check is also a served reduction
        a = numpy.arange(1, 33, dtype='f')
        a[::5] = numpy.nan
        self._dispatch_and_compare(routine, {}, a, times_called=2)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_argmax_f_order(self):
        # _J indices are C-order; the F-contiguous input goes through
        # ascontiguousarray
        a_np = testing.shaped_random((30, 40), numpy, dtype='d', seed=0)
        a = cupy.asarray(numpy.asfortranarray(a_np))
        func = _cuda_compute_reduction._cuda_compute_reduce
        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=func, times_called=1):
            result = cupy.argmax(a)
        assert int(result) == int(numpy.argmax(a_np))

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_segmented_argmin_declines(self):
        # a segmented reduction wants within-segment indices; the zip
        # provides global ones, so the resolver declines
        a_np = testing.shaped_random((20, 30), numpy, dtype='f', seed=0)
        a = cupy.asarray(a_np)
        func = _cuda_compute_reduction._cuda_compute_reduce

        def declines(*args, **kw):
            ret = func(*args, **kw)
            assert not ret
            return ret

        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=declines, times_called=1):
            result = cupy.argmin(a, axis=1)
        testing.assert_array_equal(result, numpy.argmin(a_np, axis=1))

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_segmented_min(self):
        a_np = testing.shaped_random((50, 40), numpy, dtype='d', seed=0)
        self._dispatch_and_compare('min', {'axis': 1}, a_np)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_complex_min_declines(self):
        # complex struct accumulators decline (see _try_accumulator)
        a_np = (testing.shaped_random((1000,), numpy, dtype='f', seed=0)
                + 1j).astype('F')
        a = cupy.asarray(a_np)
        func = _cuda_compute_reduction._cuda_compute_reduce

        def declines(*args, **kw):
            ret = func(*args, **kw)
            assert not ret
            return ret

        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._cuda_compute_reduce',
                wraps=declines, times_called=1):
            result = cupy.min(a)
        assert complex(result) == complex(numpy.min(a_np))

    @pytest.mark.parametrize(('routine', 'axis'), [
        ('sum', 1), ('sum', -1), ('prod', 1), ('all', 1), ('nansum', 1)])
    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_segmented_routines(self, routine, axis):
        dt = '?' if routine == 'all' else 'd'
        a = testing.shaped_random((100, 50), numpy, dtype=dt, seed=0)
        if routine == 'nansum':
            a[::7] = numpy.nan
        self._dispatch_and_compare(routine, {'axis': axis}, a)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_segmented_3d_trailing_axes(self):
        a = testing.shaped_random((8, 9, 10), numpy, dtype='d', seed=0)
        self._dispatch_and_compare('sum', {'axis': (1, 2)}, a)

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_annotated_ops_use_no_raw_op(self):
        a = cupy.ones((1000,), dtype='f')
        func = _cuda_compute_reduction._make_raw_op
        with testing.AssertFunctionIsCalled(
                'cupy._core._cuda_compute_reduction._make_raw_op',
                wraps=func, times_called=0):
            a.sum()
            a.prod()


class TestCudaComputeReductionFallback(CudaComputeReductionTestBase):

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_fallbacks(self):
        a = cupy.ones((1000,), dtype='f')
        func_name = ('cupy._core._cuda_compute_reduction'
                     '._cuda_compute_reduce')
        func = _cuda_compute_reduction._cuda_compute_reduce
        _accelerator.set_reduction_accelerators([])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # disabled
            a.sum()
