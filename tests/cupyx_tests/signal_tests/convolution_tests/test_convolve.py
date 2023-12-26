from cupy import testing
from cupyx import signal


class TestConvolve1d3o:

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=2e-3)
    def test_convolve1d3o(self, dtype, xp):
        a = testing.shaped_random((200,), xp=xp, dtype=dtype, scale=2) - 1
        b = testing.shaped_random((50, 50, 50), xp=xp, dtype=dtype, scale=2) - 1
        if xp is cupy:
            return signal.convolve1d3o(a, b)
        elif xp is numpy:
            s = numpy.dtype(dtype).itemsize
            from numpy.lib.stride_tricks import as_strided
            X = numpy.flip(as_strided(a, (151, 50, 50, 50), (s, s, 0, 0)), 1)
            Y = numpy.flip(as_strided(a, (151, 50, 50, 50), (s, 0, s, 0)), 2)
            Z = numpy.flip(as_strided(a, (151, 50, 50, 50), (s, 0, 0, s)), 3)
            return (X * Y * Z * b).sum(axis=(1, 2, 3))
        else:
            assert False

    @testing.for_complex_dtypes()
    def test_convolve1d3o_complex(self, dtype):
        # Just check that we can call the function
        a = testing.shaped_random((200,), dtype=dtype, scale=2) - (1 + 1j)
        b = testing.shaped_random(
            (50, 50, 50), dtype=dtype, scale=2) - (1 + 1j)
        signal.convolve1d3o(a, b)
