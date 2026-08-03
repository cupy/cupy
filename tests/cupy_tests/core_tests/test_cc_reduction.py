from __future__ import annotations

import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cc_reduction


# This test class and its children below only test if the cuda.compute
# backend can be used or not; they don't verify its correctness as it's
# already extensively covered by existing tests (run with
# CUPY_ACCELERATORS=cuda_compute).
class CcReductionTestBase:

    @pytest.fixture(autouse=True)
    def configure(self):
        if _cc_reduction.cuda_compute is None:
            pytest.skip('cuda.compute (cuda-cccl) not found')

        self.can_use = _cc_reduction._can_use_cc_reduction

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

    def _test_can_use(
            self, name, in_dtype, out_dtype, in_shape, out_axis, expected):
        in_args = [cupy.empty(in_shape, dtype=in_dtype)]
        out_args = [cupy.empty((), dtype=out_dtype)]
        result = self.can_use(name, in_args, out_args, out_axis)
        assert result is expected


@pytest.mark.parametrize(('in_dtype', 'out_dtype'), [
    ('?', 'q'), ('b', 'q'), ('i', 'q'), ('B', 'Q'),   # promoting rows
    ('q', 'q'), ('f', 'f'), ('d', 'd'), ('F', 'F'), ('D', 'D'),
    ('i', 'i'),                                       # dtype-pinned rows
])
class TestCcReductionCanUseDtypes(CcReductionTestBase):

    def test_can_use_cc_supported_dtypes(self, in_dtype, out_dtype):
        self._test_can_use(
            'cupy_sum', in_dtype, out_dtype, (100,), (), True)


class TestCcReductionCanUseMisc(CcReductionTestBase):

    def test_can_use_cc_nonsense_input1(self):
        # two inputs are not allowed
        a = cupy.empty((100,), dtype='f')
        b = cupy.empty((100,), dtype='f')
        c = cupy.empty((), dtype='f')
        assert self.can_use('cupy_sum', [a, b], [c], ()) is False

    def test_can_use_cc_nonsense_input2(self):
        # only the sum routines are supported
        self._test_can_use('cupy_max', 'f', 'f', (100,), (), False)
        self._test_can_use('my_kernel', 'f', 'f', (100,), (), False)

    def test_can_use_cc_axis_reduction(self):
        # non-empty out_axis (axis reduction) needs a segmented reduce
        a = cupy.empty((10, 20), dtype='f')
        b = cupy.empty((10,), dtype='f')
        assert self.can_use('cupy_sum', [a], [b], (0,)) is False

    def test_can_use_cc_float16(self):
        # must fall back: cc would accumulate in __half; the type table
        # requires a float accumulator (reduce_type 'float')
        self._test_can_use('cupy_sum', 'e', 'e', (100,), (), False)
        self._test_can_use('cupy_sum', 'f', 'e', (100,), (), False)

    def test_can_use_cc_mixed_complex(self):
        # real input with complex output (or vice versa) would mix
        # OpKind dispatch with a complex accumulator
        self._test_can_use('cupy_sum', 'f', 'F', (100,), (), False)
        self._test_can_use('cupy_sum', 'F', 'f', (100,), (), False)

    def test_can_use_cc_non_contiguous(self):
        a = cupy.empty((200,), dtype='f')[::2]
        b = cupy.empty((), dtype='f')
        assert self.can_use('cupy_sum', [a], [b], ()) is False

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_can_use_accelerator_set_unset(self):
        a = cupy.ones((1000,), dtype='f')
        func_name = 'cupy._core._cc_reduction._cc_device_sum'
        func = _cc_reduction._cc_device_sum
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            a.sum()
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # axis: falls back
            a.reshape(10, 100).sum(axis=0)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # fp16: falls back
            cupy.ones((100,), dtype='e').sum()

        _accelerator.set_reduction_accelerators([])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # disabled
            a.sum()
