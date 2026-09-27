from __future__ import annotations

import numpy
import pytest

from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cuda_compute_common


_CC_BINCOUNT = ('cupy._statistics.histogram.'
                '_cuda_compute_histogram.cuda_compute_bincount')


# These tests check that cupy.bincount reaches the cuda.compute backend.
# Correctness of the result is covered by test_histogram.py run with
# CUPY_ACCELERATORS=cuda_compute.
@pytest.fixture(autouse=True)
def use_cuda_compute_accelerator():
    if _cuda_compute_common._get_cuda_compute() is None:
        pytest.skip('cuda.compute (cuda-cccl) not found')

    old_routine_accelerators = _accelerator.get_routine_accelerators()
    _accelerator.set_routine_accelerators(['cuda_compute'])
    yield
    _accelerator.set_routine_accelerators(old_routine_accelerators)


@pytest.mark.thread_unsafe(
    reason="AssertFunctionIsCalled and accelerator mutation.")
class TestCudaComputeBincount:

    @testing.for_int_dtypes('dtype', no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_bincount(self, xp, dtype):
        dtype = xp.dtype(dtype)
        if dtype == xp.uint64:
            pytest.skip("only numpy raises exception on uint64 input")
        max_val = xp.iinfo(dtype).max if dtype.itemsize < 4 else 65536
        x = xp.arange(max_val, dtype=dtype)

        if xp is numpy:
            return xp.bincount(x)

        with testing.AssertFunctionIsCalled(_CC_BINCOUNT):
            xp.bincount(x)
        return xp.bincount(x)

    @testing.numpy_cupy_array_equal()
    def test_bincount_bool(self, xp):
        x = xp.array([True, False, True, True, False])

        if xp is numpy:
            return xp.bincount(x)

        with testing.AssertFunctionIsCalled(_CC_BINCOUNT):
            xp.bincount(x)
        return xp.bincount(x)

    @testing.numpy_cupy_array_equal()
    def test_bincount_minlength(self, xp):
        x = xp.arange(1000, dtype='i') % 32

        if xp is numpy:
            return xp.bincount(x, minlength=100)

        with testing.AssertFunctionIsCalled(_CC_BINCOUNT):
            xp.bincount(x, minlength=100)
        return xp.bincount(x, minlength=100)

    @testing.numpy_cupy_array_equal()
    def test_bincount_strided(self, xp):
        # a strided view is copied to a contiguous array before use
        x = (xp.arange(2000, dtype='i') % 32)[::2]

        if xp is numpy:
            return xp.bincount(x)

        with testing.AssertFunctionIsCalled(_CC_BINCOUNT):
            xp.bincount(x)
        return xp.bincount(x)
