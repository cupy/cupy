import numpy
import pytest

import cupy
from cupy import testing
from cupyx import signal


class TestConvolve1d2o:

    def _convolve1d2o(self, in1, in2):
        dtype = in1.dtype
        W, H = in2.shape
        size = in1.shape[0] - max(W, H) + 1
        s = numpy.dtype(dtype).itemsize
        from numpy.lib.stride_tricks import as_strided
        X = as_strided(in1, (size, W), (s, s))[:, ::-1]
        Y = as_strided(in1, (size, H), (s, s))[:, ::-1]
        return numpy.einsum('ix,iy,xy->i', X, Y, in2)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=2e-3)
    @pytest.mark.parametrize('shape', [(50, 50), (40, 60)])
    def test_convolve1d2o(self, dtype, xp, shape):
        a = testing.shaped_random((200,), xp=xp, dtype=dtype, scale=2) - 1
        b = testing.shaped_random(shape, xp=xp, dtype=dtype, scale=2) - 1
        if xp is cupy:
            return signal.convolve1d2o(a, b)
        else:
            assert xp is numpy
            return self._convolve1d2o(a, b)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=2e-3)
    @pytest.mark.parametrize('shape', [(50, 50), (40, 60)])
    def test_convolve1d2o_complex(self, dtype, xp, shape):
        # Just check that we can call the function
        a = testing.shaped_random(
            (200,), xp=xp, dtype=dtype, scale=2) - (1 + 1j)
        b = testing.shaped_random(
            shape, xp=xp, dtype=dtype, scale=2) - (1 + 1j)
        if xp is cupy:
            return signal.convolve1d2o(a, b)
        else:
            assert xp is numpy
            return self._convolve1d2o(a, b)


class TestConvolve1d3o:

    def _convolve1d3o(self, in1, in2):
        dtype = in1.dtype
        W, H, D = in2.shape
        size = in1.shape[0] - max(W, H, D) + 1
        s = numpy.dtype(dtype).itemsize
        from numpy.lib.stride_tricks import as_strided
        X = as_strided(in1, (size, W), (s, s))[:, ::-1]
        Y = as_strided(in1, (size, H), (s, s))[:, ::-1]
        Z = as_strided(in1, (size, D), (s, s))[:, ::-1]
        return numpy.einsum('ix,iy,iz,xyz->i', X, Y, Z, in2)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=2e-3)
    @pytest.mark.parametrize('shape', [(50, 50, 50), (40, 50, 60)])
    def test_convolve1d3o(self, dtype, xp, shape):
        a = testing.shaped_random((200,), xp=xp, dtype=dtype, scale=2) - 1
        b = testing.shaped_random(shape, xp=xp, dtype=dtype, scale=2) - 1
        if xp is cupy:
            return signal.convolve1d3o(a, b)
        else:
            assert xp is numpy
            return self._convolve1d3o(a, b)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=2e-3)
    @pytest.mark.parametrize('shape', [(50, 50, 50), (40, 50, 60)])
    def test_convolve1d3o_complex(self, dtype, xp, shape):
        # Just check that we can call the function
        a = testing.shaped_random(
            (200,), xp=xp, dtype=dtype, scale=2) - (1 + 1j)
        b = testing.shaped_random(
            shape, xp=xp, dtype=dtype, scale=2) - (1 + 1j)
        if xp is cupy:
            return signal.convolve1d3o(a, b)
        else:
            assert xp is numpy
            return self._convolve1d3o(a, b)
