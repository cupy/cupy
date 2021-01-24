import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
@pytest.mark.skipif(cuda.runtime.is_hip,
                    reason='HIP does not support this')
@pytest.mark.skipif(cuda.driver.get_build_version() < 10010,
                    reason='Only CUDA 10.1+ supports this')
class TestGraph(unittest.TestCase):

    def _helper1(self, a):
        # this tests ufuncs involving simple arithmetic
        a = a + 3
        a = a * 7.6
        return a**2

    def _helper2(self, a):
        # this tests ufuncs involving math API calls
        a = 3 * cupy.sin(a)
        return cupy.sqrt(a)

    def _helper3(self, a):
        # this tests CUDA library calls
        a = a * cupy.fft.fft(a)
        return cupy.fft.ifft(a)

    def _helper4(self, a):
        # this tests a common pattern in CuPy internal in which the host
        # operation depends on intermediate outcome on GPU (and thus requires
        # synchronization); here, we are protected by some (async) CUDA API
        # calls as well as launching of custom kernels
        result = cupy.zeros((1,), dtype=cupy.int32)
        if a.sum() > 0:  # synchronize!
            result += 1
        if a[-1] >= 0:  # synchronize!
            result += 2
        return result

    def test_stream_capture(self):
        s = cupy.cuda.Stream(non_blocking=True)

        for n in range(4):
            func = getattr(self, '_helper{}'.format(n+1))
            a = cupy.random.random((100,))

            with s:
                s.begin_capture()
                out1 = func(a)
                g = s.end_capture()
            g.launch()
            s.synchronize()

            out2 = func(a)
            testing.assert_array_equal(out1, out2)

    def test_stream_is_capturing(self):
        s = cupy.cuda.Stream(non_blocking=True)
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            assert s.is_capturing()
            assert not cuda.Stream.null.is_capturing()
            b = a * 3
            g = s.end_capture()
        assert not s.is_capturing()
        assert not cuda.Stream.null.is_capturing()

        # check the graph integraty
        g.launch()
        s.synchronize()
        testing.assert_array_equal(b, 3 * a)

    def test_null_stream_cannot_capture(self):
        s = cupy.cuda.Stream(non_blocking=False)
        a = cupy.random.random((100,))

        with s:
            s.begin_capture()
            b = a + 4
            assert s.is_capturing()
            # cudaStreamLegacy is unhappy when a blocking stream is capturing
            with pytest.raises(cuda.runtime.CUDARuntimeError) as e:
                assert cuda.Stream.null.is_capturing()
            assert 'cudaErrorStreamCaptureImplicit' in str(e.value)
            g = s.end_capture()
        assert not s.is_capturing()
        assert not cuda.Stream.null.is_capturing()

        # check the graph integraty
        g.launch()
        s.synchronize()
        testing.assert_array_equal(b, a + 4)
