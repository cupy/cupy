from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cuda_compute_common
from cupy._core import _cuda_compute_scan


# This test class and its children below only test if the cuda.compute
# backend can be used or not; they don't verify its correctness as it's
# already extensively covered by existing tests (run with
# CUPY_ACCELERATORS=cuda_compute).
class CudaComputeScanTestBase:

    @pytest.fixture(autouse=True)
    def configure(self):
        if _cuda_compute_common._get_cuda_compute() is None:
            pytest.skip('cuda.compute (cuda-cccl) not found')

        self.supports_dtype = _cuda_compute_scan._supports_dtype

        self.old_routine_accelerators = (
            _accelerator.get_routine_accelerators())
        _accelerator.set_routine_accelerators(['cuda_compute'])
        yield
        _accelerator.set_routine_accelerators(self.old_routine_accelerators)


class TestCudaComputeScanDtypes(CudaComputeScanTestBase):

    # scan_core only promotes when dtype=None and out=None, so an
    # explicit dtype=/out= reaches the accelerator unpromoted
    @testing.for_all_dtypes(no_bool=True)
    def test_supported_dtypes(self, dtype):
        assert self.supports_dtype(numpy.dtype(dtype)) is True


class TestCudaComputeScanMisc(CudaComputeScanTestBase):

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
