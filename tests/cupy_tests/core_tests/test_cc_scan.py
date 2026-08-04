from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cc_scan


# This test class and its children below only test if the cuda.compute
# backend can be used or not; they don't verify its correctness as it's
# already extensively covered by existing tests (run with
# CUPY_ACCELERATORS=cuda_compute).
class CcScanTestBase:

    @pytest.fixture(autouse=True)
    def configure(self):
        if _cc_scan.cuda_compute is None:
            pytest.skip('cuda.compute (cuda-cccl) not found')

        self.supports_dtype = _cc_scan._supports_dtype

        self.old_routine_accelerators = (
            _accelerator.get_routine_accelerators())
        _accelerator.set_routine_accelerators(['cuda_compute'])
        yield
        _accelerator.set_routine_accelerators(self.old_routine_accelerators)


class TestCcScanCanUseDtypes(CcScanTestBase):

    # scan_core promotes before dispatch, so only post-promotion dtypes
    # reach the accelerator (e.g. int32 input arrives as int64)
    @pytest.mark.parametrize('dtype', [
        'l', 'L', 'q', 'Q', 'e', 'f', 'd', 'F', 'D'])
    def test_supported_dtypes(self, dtype):
        assert self.supports_dtype(numpy.dtype(dtype)) is True


class TestCcScanCanUseMisc(CcScanTestBase):

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerator mutation.")
    def test_can_use_accelerator_set_unset(self):
        a = cupy.ones((1000,), dtype='f')

        func_name = 'cupy._core._cc_scan._cc_scan_arrays'
        func = _cc_scan._cc_scan_arrays
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):
            cupy.cumsum(a)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):  # promoting int
            cupy.cumsum(cupy.ones((1000,), dtype='i'))
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):  # bool promotes too
            cupy.cumsum(cupy.ones((1000,), dtype='?'))
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):  # strided: copied
            cupy.cumsum(cupy.ones((2000,), dtype='f')[::2])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # axis: falls back
            cupy.cumsum(a.reshape(10, 100), axis=0)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):  # cumprod
            cupy.cumprod(a)

        _accelerator.set_routine_accelerators([])
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # disabled
            cupy.cumsum(a)
