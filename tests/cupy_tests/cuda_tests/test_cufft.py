import pickle
import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._fft import _convert_fft_type

from ..fft_tests.test_fft import (multi_gpu_config, _skip_multi_gpu_bug)


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cufft.CuFFTError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


# This class tests multi-GPU Plan1d with data sitting on host.
# Internally, we use cuFFT's data transfer API to ensure the
# data is in order.

@testing.parameterize(*testing.product({
    'shape': [(64,), (4, 16), (128,), (8, 32)],
}))
@testing.multi_gpu(2)
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason='not supported by hipFFT')
class TestMultiGpuPlan1dNumPy(unittest.TestCase):

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    def test_fft(self, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, numpy, dtype)

        if len(self.shape) == 1:
            batch = 1
            nx = self.shape[0]
        elif len(self.shape) == 2:
            batch = self.shape[0]
            nx = self.shape[1]

        # compute via cuFFT
        cufft_type = _convert_fft_type(a.dtype, 'C2C')
        plan = cufft.Plan1d(nx, cufft_type, batch, devices=config._devices)
        out_cp = numpy.empty_like(a)
        plan.fft(a, out_cp, cufft.CUFFT_FORWARD)

        out_np = numpy.fft.fft(a)
        # np.fft.fft alway returns np.complex128
        if dtype is numpy.complex64:
            out_np = out_np.astype(dtype)

        assert numpy.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)

        # compute it again to ensure Plan1d's internal state is reset
        plan.fft(a, out_cp, cufft.CUFFT_FORWARD)

        assert numpy.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)

    @multi_gpu_config(gpu_configs=[[0, 1], [1, 0]])
    @testing.for_complex_dtypes()
    def test_ifft(self, dtype):
        _skip_multi_gpu_bug(self.shape, self.gpus)

        a = testing.shaped_random(self.shape, numpy, dtype)

        if len(self.shape) == 1:
            batch = 1
            nx = self.shape[0]
        elif len(self.shape) == 2:
            batch = self.shape[0]
            nx = self.shape[1]

        # compute via cuFFT
        cufft_type = _convert_fft_type(a.dtype, 'C2C')
        plan = cufft.Plan1d(nx, cufft_type, batch, devices=config._devices)
        out_cp = numpy.empty_like(a)
        plan.fft(a, out_cp, cufft.CUFFT_INVERSE)
        # normalization
        out_cp /= nx

        out_np = numpy.fft.ifft(a)
        # np.fft.fft alway returns np.complex128
        if dtype is numpy.complex64:
            out_np = out_np.astype(dtype)

        assert numpy.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)

        # compute it again to ensure Plan1d's internal state is reset
        plan.fft(a, out_cp, cufft.CUFFT_INVERSE)
        # normalization
        out_cp /= nx

        assert numpy.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)


# This class contains a few simple tests for the new XtPlanNd wrapper. A full-
# scale test will be done (with little cost) when this low-level API is
# integrated with the high-level APIs such as cupy.fft.fft() (see #4406), so
# don't make our life too hard here. Note that we don't have cupy.complex32
# yet, so its dtype is represented by a single character 'E' (in contrast to
# 'e' for float16) for the time being (see #3370).

@testing.parameterize(*testing.product({
    'shape': [(4, 16), (4, 16, 16)],
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason='not supported by hipFFT')
class TestXtPlanNd(unittest.TestCase):

    @testing.for_complex_dtypes()
    def test_forward_fft(self, dtype):
        t = dtype
        idtype = odtype = edtype = cupy.dtype(t)
        shape = self.shape
        length = cupy._core.internal.prod(shape[1:])

        a = testing.shaped_random(shape, cupy, dtype)
        out = cupy.empty_like(a)

        plan = cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, length, idtype,
                              shape[1:], 1, length, odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)
        plan.fft(a, out, cufft.CUFFT_FORWARD)

        if len(shape) <= 2:
            out_cp = cupy.fft.fft(a)
        else:
            out_cp = cupy.fft.fftn(a, axes=(-1, -2))
        testing.assert_allclose(out, out_cp)

    @testing.for_complex_dtypes()
    def test_backward_fft(self, dtype):
        t = dtype
        idtype = odtype = edtype = cupy.dtype(t)
        shape = self.shape
        length = cupy._core.internal.prod(shape[1:])

        a = testing.shaped_random(shape, cupy, dtype)
        out = cupy.empty_like(a)

        plan = cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, length, idtype,
                              shape[1:], 1, length, odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)
        plan.fft(a, out, cufft.CUFFT_INVERSE)

        if len(shape) <= 2:
            out_cp = cupy.fft.ifft(a)
        else:
            out_cp = cupy.fft.ifftn(a, axes=(-1, -2))
        testing.assert_allclose(out/length, out_cp)

    @pytest.mark.skipif(int(cupy.cuda.device.get_compute_capability()) < 53,
                        reason='half-precision complex FFT is not supported')
    def test_forward_fft_complex32(self):
        t = 'E'
        idtype = odtype = edtype = t
        old_shape = self.shape
        shape = list(self.shape)
        shape[-1] = 2*shape[-1]
        shape = tuple(shape)

        a = testing.shaped_random(shape, cupy, cupy.float16)
        out = cupy.empty_like(a)

        shape = old_shape
        length = cupy._core.internal.prod(shape[1:])
        plan = cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, length, idtype,
                              shape[1:], 1, length, odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)
        plan.fft(a, out, cufft.CUFFT_FORWARD)

        a_cp = a.astype(cupy.float32)  # upcast
        a_cp = a_cp.view(cupy.complex64)
        if len(shape) <= 2:
            out_cp = cupy.fft.fft(a_cp)
        else:
            out_cp = cupy.fft.fftn(a_cp, axes=(-1, -2))
        out_cp = out_cp.view(cupy.float32)
        out_cp = out_cp.astype(cupy.float16)  # downcast

        # We shouldn't expect getting high accuracy, as both computations have
        # precision losses...
        testing.assert_allclose(out, out_cp, rtol=1E-1, atol=1E-1)

    @pytest.mark.skipif(int(cupy.cuda.device.get_compute_capability()) < 53,
                        reason='half-precision complex FFT is not supported')
    def test_backward_fft_complex32(self):
        t = 'E'
        idtype = odtype = edtype = t
        old_shape = self.shape
        shape = list(self.shape)
        shape[-1] = 2*shape[-1]
        shape = tuple(shape)

        a = testing.shaped_random(shape, cupy, cupy.float16)
        out = cupy.empty_like(a)

        shape = old_shape
        length = cupy._core.internal.prod(shape[1:])
        plan = cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, length, idtype,
                              shape[1:], 1, length, odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)
        plan.fft(a, out, cufft.CUFFT_INVERSE)

        a_cp = a.astype(cupy.float32)  # upcast
        a_cp = a_cp.view(cupy.complex64)
        if len(shape) <= 2:
            out_cp = cupy.fft.ifft(a_cp)
        else:
            out_cp = cupy.fft.ifftn(a_cp, axes=(-1, -2))
        out_cp = out_cp.view(cupy.float32)
        out_cp = out_cp.astype(cupy.float16)  # downcast

        # We shouldn't expect getting high accuracy, as both computations have
        # precision losses...
        testing.assert_allclose(out/length, out_cp, rtol=1E-1, atol=1E-1)
