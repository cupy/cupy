import pickle
import unittest

import numpy

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
