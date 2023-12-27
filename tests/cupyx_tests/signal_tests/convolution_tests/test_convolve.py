import numpy
import pytest

import cupy
from cupy import testing
from cupyx import signal


def _convolve1d3o(in1, in2):
    dtype = in1.dtype
    ker_shape = in2.shape
    out_dim = in1.shape[0] - max(ker_shape) + 1
    s = numpy.dtype(dtype).itemsize
    from numpy.lib.stride_tricks import as_strided
    X = numpy.flip(as_strided(in1, (out_dim, *ker_shape), (s, s, 0, 0)), 1)
    Y = numpy.flip(as_strided(in1, (out_dim, *ker_shape), (s, 0, s, 0)), 2)
    Z = numpy.flip(as_strided(in1, (out_dim, *ker_shape), (s, 0, 0, s)), 3)
    return (X * Y * Z * in2).sum(axis=(1, 2, 3))


class TestConvolve1d3o:

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=2e-3)
    @pytest.mark.parametrize('shape', [(50, 50, 50), (40, 50, 60)])
    def test_convolve1d3o(self, dtype, xp, shape):
        a = testing.shaped_random((200,), xp=xp, dtype=dtype, scale=2) - 1
        b = testing.shaped_random(shape, xp=xp, dtype=dtype, scale=2) - 1
        if xp is cupy:
            return signal.convolve1d3o(a, b)
        elif xp is numpy:
            return _convolve1d3o(a, b)
        else:
            assert False

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
        elif xp is numpy:
            return _convolve1d3o(a, b)
        else:
            assert False
