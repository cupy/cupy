from cupy import testing
from cupyx import signal


class TestConvolve1d3o:

    @testing.for_float_dtypes(no_float16=True)
    def test_convolve1d3o(self, dtype):
        # Just check that we can call the function
        a = testing.shaped_random((200,), dtype=dtype, scale=2) - 1
        b = testing.shaped_random((50, 50, 50), dtype=dtype, scale=2) - 1
        signal.convolve1d3o(a, b)

    @testing.for_complex_dtypes()
    def test_convolve1d3o_complex(self, dtype):
        # Just check that we can call the function
        a = testing.shaped_random((200,), dtype=dtype, scale=2) - (1 + 1j)
        b = testing.shaped_random(
            (50, 50, 50), dtype=dtype, scale=2) - (1 + 1j)
        signal.convolve1d3o(a, b)
