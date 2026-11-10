from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupyx import signal
from cupyx.signal._convolution import _convolve


@pytest.mark.parametrize('shape', [(0, 2**31), (2**31, 0)])
def test_convolve1d2o_kernel_empty_filter(shape):
    # A zero extent makes the sum empty while a sibling extent need not fit
    # the filter's index type; narrowing it must not resurrect the sum.
    in1 = cupy.zeros(1, dtype=cupy.float32)
    in2 = cupy.empty(shape, dtype=cupy.float32)
    out = cupy.full(1, 7, dtype=cupy.float32)
    _convolve._convolve1d2o_kernel(in1, in2, *shape, out)
    assert out[0] == 0


@pytest.mark.parametrize(
    'shape', [(0, 1, 2**31), (1, 2**31, 0), (2**31, 0, 1)])
def test_convolve1d3o_kernel_empty_filter(shape):
    # A zero extent makes the sum empty while a sibling extent need not fit
    # the filter's index type; narrowing it must not resurrect the sum.
    in1 = cupy.zeros(1, dtype=cupy.float32)
    in2 = cupy.empty(shape, dtype=cupy.float32)
    out = cupy.full(1, 7, dtype=cupy.float32)
    _convolve._convolve1d3o_kernel(in1, in2, *shape, out)
    assert out[0] == 0


@pytest.mark.slow
@pytest.mark.parametrize('order', [2, 3])
def test_convolve1do_kernel_indexes_beyond_32_bits(order):
    # The `in1` subscripts follow the output index, which passes 2**31 here
    # while the filter stays small enough to be indexed with int32. Both
    # `in1` and `out` cycle through three values (as strided views, so
    # neither costs real memory), which makes an index truncated to int32
    # read the wrong element rather than aliasing onto the right one.
    as_strided = cupy.lib.stride_tricks.as_strided
    pattern = cupy.array([1, 2, 4], dtype=cupy.float32)
    out_base = cupy.zeros(pattern.size, dtype=cupy.float32)
    shape = ((2**31 + 2) // pattern.size + 1, pattern.size)
    strides = (0, pattern.itemsize)
    in1 = as_strided(pattern, shape=shape, strides=strides)
    out = as_strided(out_base, shape=shape, strides=strides)
    in2 = cupy.ones((1,) * order, dtype=cupy.float32)

    if order == 2:
        _convolve._convolve1d2o_kernel(in1, in2, *in2.shape, out)
    else:
        _convolve._convolve1d3o_kernel(in1, in2, *in2.shape, out)

    testing.assert_array_equal(out_base, pattern ** order)


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
