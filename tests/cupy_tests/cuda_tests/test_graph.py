import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
@pytest.mark.skipif(cuda.runtime.is_hip,
                    reason='HIP does not support this')
@pytest.mark.skipif(cuda.driver.get_build_version() < 10010,
                    reason='Only CUDA 11.0+ supports this')
class TestGraph(unittest.TestCase):

    def _helper1(self, a):
        a = a + 3
        a = a * 7.6
        return a**2

    def _helper2(self, a):
        a = 3 * cupy.sin(a)
        return cupy.sqrt(a)

    def _helper3(self, a):
        a = a * cupy.fft.fft(a)
        return cupy.fft.ifft(a)

    def test_stream_capture(self):
        s = cupy.cuda.Stream(non_blocking=True)

        for n in range(1, 4):
            func = getattr(self, '_helper{}'.format(n))
            a = cupy.random.random((100,))

            with s:
                s.begin_capture()
                out1 = func(a)
                g = s.end_capture()
            g.launch()
            s.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)
