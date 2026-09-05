from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cuda_compute_common


@unittest.skipUnless(
    _cuda_compute_common._get_cuda_compute() is not None,
    'cuda.compute (cuda-cccl) is not available')
class TestCudaComputeBincount(unittest.TestCase):
    @pytest.fixture(autouse=True, scope='class')
    @classmethod
    def setup(cls):
        old_accelerators = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators(['cuda_compute'])
        yield
        _accelerator.set_routine_accelerators(old_accelerators)

    @pytest.mark.thread_unsafe(reason="uses AssertFunctionIsCalled")
    @testing.for_int_dtypes('dtype', no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_bincount(self, xp, dtype):
        dtype = xp.dtype(dtype)
        max_val = xp.iinfo(dtype).max if dtype.itemsize < 4 else 65536
        if dtype == xp.uint64:
            pytest.skip("only numpy raises exception on uint64 input")
        x = xp.arange(max_val, dtype=dtype)

        if xp is numpy:
            return xp.bincount(x)

        # xp is cupy, first ensure we really use cuda.compute
        cc_func = ('cupy._statistics.histogram.'
                   '_cuda_compute_histogram.cuda_compute_bincount')
        with testing.AssertFunctionIsCalled(cc_func):
            xp.bincount(x)
        # ...then perform the actual computation
        return xp.bincount(x)

    @pytest.mark.thread_unsafe(reason="uses AssertFunctionIsCalled")
    @testing.numpy_cupy_array_equal()
    def test_bincount_minlength(self, xp):
        x = xp.arange(1000, dtype='i') % 32

        if xp is numpy:
            return xp.bincount(x, minlength=100)

        cc_func = ('cupy._statistics.histogram.'
                   '_cuda_compute_histogram.cuda_compute_bincount')
        with testing.AssertFunctionIsCalled(cc_func):
            xp.bincount(x, minlength=100)
        return xp.bincount(x, minlength=100)

    @pytest.mark.thread_unsafe(reason="uses AssertFunctionIsCalled")
    @testing.numpy_cupy_array_equal()
    def test_bincount_strided(self, xp):
        # strided input is contiguized before use
        x = (xp.arange(2000, dtype='i') % 32)[::2]

        if xp is numpy:
            return xp.bincount(x)

        cc_func = ('cupy._statistics.histogram.'
                   '_cuda_compute_histogram.cuda_compute_bincount')
        with testing.AssertFunctionIsCalled(cc_func):
            xp.bincount(x)
        return xp.bincount(x)

    @pytest.mark.thread_unsafe(reason="uses AssertFunctionIsCalled")
    def test_weighted_not_served(self):
        # weighted bincount has no cuda.compute path
        x = cupy.arange(1000, dtype='i') % 32
        cc_func = ('cupy._statistics.histogram.'
                   '_cuda_compute_histogram.cuda_compute_bincount')
        with testing.AssertFunctionIsCalled(cc_func, times_called=0):
            cupy.bincount(x, weights=cupy.ones(1000, dtype='d'))
