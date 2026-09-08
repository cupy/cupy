from __future__ import annotations

import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cuda_compute_scan


# The tests in this module only test if the cuda.compute backend can
# be used or not; they don't verify its correctness as it's already
# extensively covered by existing tests (run with
# CUPY_ACCELERATORS=cuda_compute).
@pytest.fixture(autouse=True)
def use_cuda_compute_accelerator():
    if _cuda_compute_scan._get_cuda_compute() is None:
        pytest.skip('cuda.compute (cuda-cccl) not found')

    old_routine_accelerators = _accelerator.get_routine_accelerators()
    _accelerator.set_routine_accelerators(['cuda_compute'])
    yield
    _accelerator.set_routine_accelerators(old_routine_accelerators)


class TestCudaComputeScanMisc:

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_can_use_accelerator_set_unset(self):
        a = cupy.ones((1000,), dtype='f')

        func_name = 'cupy._core._cuda_compute_scan._cuda_compute_scan_arrays'
        func = _cuda_compute_scan._cuda_compute_scan_arrays
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumsum(a)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumsum(cupy.ones((1000,), dtype='i'))
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumsum(cupy.ones((1000,), dtype='?'))
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumsum(cupy.ones((2000,), dtype='f')[::2])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumprod(a)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):
            cupy.cumsum(a.reshape(10, 100), axis=0)

        _accelerator.set_routine_accelerators([])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):
            cupy.cumsum(a)
